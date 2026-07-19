import pytest

from modules import extra_networks


def test_identical_extra_network_stacks_are_allowed():
    prompts, data = extra_networks.parse_prompts([
        'first <lora:character:1:1:role=char>',
        'second <lora:character:1:1:role=char>',
    ])
    assert prompts == ['first ', 'second ']
    assert data['lora'][0].named['role'] == 'char'


def test_different_extra_network_stacks_are_rejected():
    with pytest.raises(ValueError, match='Prompt 2 uses a different extra-network stack'):
        extra_networks.parse_prompts([
            'first <lora:character:1:1:role=char>',
            'second <lora:style:0.7:0.7:role=style>',
        ])


def test_empty_prompt_list_returns_empty_data():
    prompts, data = extra_networks.parse_prompts([])
    assert prompts == []
    assert dict(data) == {}
