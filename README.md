Improved reForge is aimed at enhancing performance and optimization. Feel free to open an issue if you have any optimization ideas.

There are also some minor config adjustments to improve generation speed, while still staying within a safe range without reducing image quality.

<hr>

- [x] Removed some unecessary features
- [x] Improved lora-block-weight model accuracy 
- [x] Added Sage Attention and Flash Attention.
  It's not really a big performance gain, but there is a noticeable speed improvement, around 0.7-0.8 it/s.<br>
  [SageAttention and FlashAttention setup](docs/sage_flash_attention_setup.md)<br>
  Refer: [https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2866](https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/2866)
