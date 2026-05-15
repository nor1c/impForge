import hashlib
import random

import gradio as gr
import torch

from modules import scripts


DEFAULT_DANBOORU_FRAMING_TAGS = """from above
from below
from side
from behind
three-quarter view
profile
cowboy shot
upper body
close-up
wide shot
looking at viewer
looking away
looking to the side
dynamic pose
sitting
standing
walking
reaching
arms up
arms behind back"""


class BatchDiversityScript(scripts.Script):
    sorting_priority = 15.8

    def title(self):
        return "Batch Diversity"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Batch Diversity", open=False):
            enabled = gr.Checkbox(
                label="Enable Batch Diversity",
                value=False,
                info="Diversifies batches without changing your prompt by default.",
            )
            mode = gr.Radio(
                label="Mode",
                choices=["Latent + seeds", "Latent only", "Latent + danbooru framing"],
                value="Latent + seeds",
            )
            strength = gr.Slider(
                label="Diversity strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.35,
                info="Higher values push initial noise further from the original seed. 0.25-0.45 is usually a safe range.",
            )
            independent_seeds = gr.Checkbox(
                label="Independent seeds",
                value=True,
                info="Uses hash-derived independent seeds for every image and breaks same-base variation mode.",
            )
            latent_jitter = gr.Checkbox(
                label="Latent noise jitter",
                value=True,
                info="Blends deterministic extra noise into each image's initial latent noise.",
            )
            spatial_shift = gr.Checkbox(
                label="Spatial latent shift",
                value=True,
                info="Rolls latent noise differently for each image to push composition without prompt changes.",
            )
            danbooru_tags_enabled = gr.Checkbox(
                label="Optional danbooru framing tags",
                value=False,
                info="Disabled by default. Only uses the tag list below; do not add lighting/background/style tags unless you want them.",
            )
            tags_per_image = gr.Slider(
                label="Danbooru tags per image",
                minimum=1,
                maximum=4,
                step=1,
                value=2,
            )
            danbooru_tag_list = gr.Textbox(
                label="Allowed danbooru framing tags",
                value=DEFAULT_DANBOORU_FRAMING_TAGS,
                lines=8,
                info="One tag per line. Default list avoids lighting, background, quality, style, and setting tags.",
            )

        return (
            enabled,
            mode,
            strength,
            independent_seeds,
            latent_jitter,
            spatial_shift,
            danbooru_tags_enabled,
            tags_per_image,
            danbooru_tag_list,
        )

    def process(
        self,
        p,
        enabled,
        mode,
        strength,
        independent_seeds,
        latent_jitter,
        spatial_shift,
        danbooru_tags_enabled,
        tags_per_image,
        danbooru_tag_list,
    ):
        if not enabled:
            return

        total = len(getattr(p, "all_prompts", []) or [])
        if total <= 1:
            return

        base_seed = int((getattr(p, "all_seeds", None) or [getattr(p, "seed", 0)])[0])
        p._batch_diversity_base_seed = base_seed
        p._batch_diversity_strength = float(strength)
        p._batch_diversity_latent_jitter = bool(latent_jitter)
        p._batch_diversity_spatial_shift = bool(spatial_shift)

        if mode in ("Latent + seeds", "Latent + danbooru framing") and independent_seeds:
            p.all_seeds = [base_seed if i == 0 else self._mixed_seed(base_seed, i, "seed") for i in range(total)]
            p.all_subseeds = [self._mixed_seed(base_seed, i, "subseed") for i in range(total)]
            if getattr(p, "subseed_strength", 0) != 0:
                p.subseed_strength = 0

        prompt_variants = None
        use_framing = mode == "Latent + danbooru framing" and danbooru_tags_enabled
        if use_framing:
            tags = self._parse_tags(danbooru_tag_list)
            if tags:
                prompts = list(p.all_prompts)
                prompt_variants = []
                for i, prompt in enumerate(prompts):
                    chosen = self._pick_tags(tags, p.all_seeds[i], i, int(tags_per_image))
                    prompts[i] = self._append_tags(prompt, chosen)
                    prompt_variants.append(", ".join(chosen))
                p.all_prompts = prompts

                if hasattr(p, "all_hr_prompts") and p.all_hr_prompts:
                    p.all_hr_prompts = [self._append_tags(prompt, prompt_variants[i].split(", ")) for i, prompt in enumerate(p.all_hr_prompts)]

        params = {
            "Batch Diversity": mode,
            "Batch Diversity strength": round(float(strength), 3),
            "Batch Diversity independent seeds": bool(independent_seeds and mode in ("Latent + seeds", "Latent + danbooru framing")),
            "Batch Diversity latent jitter": bool(latent_jitter),
            "Batch Diversity spatial shift": bool(spatial_shift),
        }
        if prompt_variants:
            params["Batch Diversity danbooru tags"] = prompt_variants
        p.extra_generation_params.update(params)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        if not getattr(p, "_batch_diversity_latent_jitter", False) and not getattr(p, "_batch_diversity_spatial_shift", False):
            return

        noise = kwargs.get("noise", None)
        if noise is None or not torch.is_tensor(noise) or noise.ndim < 3:
            return

        strength = float(getattr(p, "_batch_diversity_strength", 0.0))
        if strength <= 0:
            return

        seeds = list(getattr(p, "seeds", []) or [])
        if not seeds:
            return

        batch = noise.detach().clone()
        mixed = []
        for i, sample in enumerate(batch):
            seed = int(seeds[min(i, len(seeds) - 1)])
            varied = sample

            if getattr(p, "_batch_diversity_spatial_shift", False):
                varied = self._roll_latent(varied, seed, i)

            if getattr(p, "_batch_diversity_latent_jitter", False):
                extra = self._randn_like(sample, self._mixed_seed(seed, i, "latent"))
                varied = varied * (1.0 - strength) + extra * strength
            else:
                varied = sample * (1.0 - strength) + varied * strength

            mixed.append(self._match_stats(varied, sample))

        p.modified_noise = torch.stack(mixed, dim=0)

    @staticmethod
    def _mixed_seed(seed, index, salt):
        data = f"batch-diversity:{salt}:{int(seed)}:{int(index)}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little") % 4294967294

    @staticmethod
    def _parse_tags(raw):
        tags = []
        seen = set()
        for line in (raw or "").replace(";", "\n").splitlines():
            tag = line.strip().strip(",")
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags

    @staticmethod
    def _pick_tags(tags, seed, index, count):
        count = max(1, min(int(count), len(tags)))
        gen = random.Random(BatchDiversityScript._mixed_seed(seed, index, "tags"))
        return gen.sample(tags, count)

    @staticmethod
    def _append_tags(prompt, tags):
        tags = [tag for tag in tags if tag]
        if not tags:
            return prompt
        suffix = ", ".join(tags)
        prompt = prompt or ""
        return f"{prompt}, {suffix}" if prompt.strip() else suffix

    @staticmethod
    def _roll_latent(sample, seed, index):
        if sample.ndim < 3:
            return sample

        height = sample.shape[-2]
        width = sample.shape[-1]
        if height <= 1 or width <= 1:
            return sample

        gen = random.Random(BatchDiversityScript._mixed_seed(seed, index, "roll"))
        shift_y = gen.randrange(0, height)
        shift_x = gen.randrange(0, width)
        return torch.roll(sample, shifts=(shift_y, shift_x), dims=(-2, -1))

    @staticmethod
    def _randn_like(sample, seed):
        try:
            generator = torch.Generator(device=sample.device).manual_seed(int(seed))
            return torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            return torch.randn(sample.shape, device="cpu", dtype=sample.dtype, generator=generator).to(sample.device)

    @staticmethod
    def _match_stats(varied, reference):
        dims = tuple(range(1, varied.ndim)) if varied.ndim > 1 else None
        if dims is None:
            return varied

        ref_mean = reference.mean(dim=dims, keepdim=True)
        ref_std = reference.std(dim=dims, keepdim=True).clamp_min(1e-6)
        var_mean = varied.mean(dim=dims, keepdim=True)
        var_std = varied.std(dim=dims, keepdim=True).clamp_min(1e-6)
        return (varied - var_mean) / var_std * ref_std + ref_mean
