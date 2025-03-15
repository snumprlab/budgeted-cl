import torch


def generate_llava_blocks():
    ### Vision Towers ###
    num_vision_blocks = 24
    vision_blocks = []
    vision_prefix = "base_model.model.model.vision_tower.vision_tower.vision_model"
    vision_blocks.append(f"{vision_prefix}.embeddings.patch_embedding")
    for num_vision_block in range(num_vision_blocks):
        for i in range(24):
            group = [f"{vision_prefix}.encoder.layers.{i}.self_attn.k_proj", 
                    f"{vision_prefix}.encoder.layers.{i}.self_attn.v_proj", 
                    f"{vision_prefix}.encoder.layers.{i}.self_attn.q_proj", 
                    f"{vision_prefix}.encoder.layers.{i}.self_attn.out_proj",
                    f"{vision_prefix}.encoder.layers.{i}.mlp.fc1",
                    f"{vision_prefix}.encoder.layers.{i}.mlp.fc2"]
            vision_blocks.extend(group)

    projection_blocks = ["base_model.model.model.mm_projector.0", "base_model.model.model.mm_projector.2"]

    ### LLM part
    num_llm_blocks = 32
    attention_block_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_proj_block_types = ["gate_proj", "up_proj", "down_proj"]
    llm_blocks = []
    for num_block in range(num_llm_blocks):
        blocks = []
        for attention_block_type in attention_block_types:
            blocks.extend([
                # f"base_model.model.model.layers.{num_block}.self_attn.{attention_block_type}",
                # f"base_model.model.model.layers.{num_block}.self_attn.{attention_block_type}.base_layer",
                f"base_model.model.model.layers.{num_block}.self_attn.{attention_block_type}.lora_A.default",
                f"base_model.model.model.layers.{num_block}.self_attn.{attention_block_type}.lora_B.default"
                ])
        for mlp_proj_block_type in mlp_proj_block_types:
            blocks.extend([
                # f"base_model.model.model.layers.0.mlp.{mlp_proj_block_type}",
                # f"base_model.model.model.layers.0.mlp.{mlp_proj_block_type}.base_layer",
                f"base_model.model.model.layers.0.mlp.{mlp_proj_block_type}.lora_A.default",
                f"base_model.model.model.layers.0.mlp.{mlp_proj_block_type}.lora_B.default"
                ])
        llm_blocks.append(blocks)
    # llm_blocks.append(["base_model.model.lm_head"])

    blocks = []
    # blocks.append(vision_blocks)
    # blocks.append(projection_blocks)
    blocks.extend(llm_blocks)

    return blocks

MODEL_BLOCK_DICT = {
    'resnet18': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group4.blocks.block0'],
                 ['group4.blocks.block1'],
                 ['pool', 'fc']],
    
    'resnet32': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group1.blocks.block2'],
                 ['group1.blocks.block3'],
                 ['group1.blocks.block4'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group2.blocks.block2'],
                 ['group2.blocks.block3'],
                 ['group2.blocks.block4'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group3.blocks.block2'],
                 ['group3.blocks.block3'],
                 ['group3.blocks.block4'],
                 ['pool', 'fc']],
    
    'vit': [['vit_model.patch_embed', 'vit_model.pos_drop', 'vit_model.patch_drop', 'vit_model.norm_pre'], 
            ['vit_model.blocks.0'], 
            ['vit_model.blocks.1'], 
            ['vit_model.blocks.2'], 
            ['vit_model.blocks.3'], 
            ['vit_model.blocks.4'], 
            ['vit_model.blocks.5'], 
            ['vit_model.blocks.6'], 
            ['vit_model.blocks.7'], 
            ['vit_model.blocks.8'], 
            ['vit_model.blocks.9'], 
            ['vit_model.blocks.10'], 
            ['vit_model.blocks.11'], 
            ['vit_model.norm', 'vit_model.fc_norm', 'vit_model.head_drop', 'head']],
    
    # 'llava': []
    'llava': generate_llava_blocks()
}

MODEL_TRAINABLE_BLOCK_DICT = {
    'llava': [['base_model.model.model.mm_projector.0', 'base_model.model.model.mm_projector.2'],
    ['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.0.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.0.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.0.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.0.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.0.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.0.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.0.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.0.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.0.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.0.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.0.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.0.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.0.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.1.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.1.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.1.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.1.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.1.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.1.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.1.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.1.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.1.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.1.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.1.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.1.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.1.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.1.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.2.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.2.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.2.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.2.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.2.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.2.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.2.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.2.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.2.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.2.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.2.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.2.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.2.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.2.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.3.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.3.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.3.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.3.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.3.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.3.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.3.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.3.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.3.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.3.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.3.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.3.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.3.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.3.mlp.down_proj.lora_B.default'],
    ['base_model.model.model.layers.4.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.4.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.4.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.4.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.4.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.4.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.4.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.4.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.4.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.4.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.4.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.4.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.4.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.4.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.5.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.5.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.5.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.5.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.5.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.5.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.5.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.5.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.5.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.5.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.5.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.5.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.5.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.5.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.6.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.6.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.6.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.6.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.6.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.6.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.6.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.6.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.6.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.6.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.6.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.6.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.6.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.6.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.7.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.7.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.7.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.7.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.7.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.7.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.7.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.7.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.7.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.7.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.7.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.7.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.7.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.7.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.8.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.8.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.8.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.8.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.8.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.8.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.8.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.8.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.8.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.8.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.8.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.8.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.8.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.8.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.9.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.9.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.9.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.9.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.9.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.9.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.9.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.9.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.9.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.9.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.9.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.9.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.9.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.9.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.10.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.10.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.10.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.10.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.10.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.10.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.10.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.10.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.10.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.10.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.10.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.10.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.10.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.10.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.11.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.11.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.11.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.11.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.11.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.11.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.11.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.11.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.11.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.11.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.11.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.11.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.11.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.11.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.12.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.12.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.12.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.12.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.12.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.12.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.12.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.12.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.12.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.12.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.12.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.12.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.12.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.12.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.13.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.13.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.13.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.13.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.13.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.13.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.13.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.13.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.13.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.13.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.13.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.13.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.13.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.13.mlp.down_proj.lora_B.default'],
    ['base_model.model.model.layers.14.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.14.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.14.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.14.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.14.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.14.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.14.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.14.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.14.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.14.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.14.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.14.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.14.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.14.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.15.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.15.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.15.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.15.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.15.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.15.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.15.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.15.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.15.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.15.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.15.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.15.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.15.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.15.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.16.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.16.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.16.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.16.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.16.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.16.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.16.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.16.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.16.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.16.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.16.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.16.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.16.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.16.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.17.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.17.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.17.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.17.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.17.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.17.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.17.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.17.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.17.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.17.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.17.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.17.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.17.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.17.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.18.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.18.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.18.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.18.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.18.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.18.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.18.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.18.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.18.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.18.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.18.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.18.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.18.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.18.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.19.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.19.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.19.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.19.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.19.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.19.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.19.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.19.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.19.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.19.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.19.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.19.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.19.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.19.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.20.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.20.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.20.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.20.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.20.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.20.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.20.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.20.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.20.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.20.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.20.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.20.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.20.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.20.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.21.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.21.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.21.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.21.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.21.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.21.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.21.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.21.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.21.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.21.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.21.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.21.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.21.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.21.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.22.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.22.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.22.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.22.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.22.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.22.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.22.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.22.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.22.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.22.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.22.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.22.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.22.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.22.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.23.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.23.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.23.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.23.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.23.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.23.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.23.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.23.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.23.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.23.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.23.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.23.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.23.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.23.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.24.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.24.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.24.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.24.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.24.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.24.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.24.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.24.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.24.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.24.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.24.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.24.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.24.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.24.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.25.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.25.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.25.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.25.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.25.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.25.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.25.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.25.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.25.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.25.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.25.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.25.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.25.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.25.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.26.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.26.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.26.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.26.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.26.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.26.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.26.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.26.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.26.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.26.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.26.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.26.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.26.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.26.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.27.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.27.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.27.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.27.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.27.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.27.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.27.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.27.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.27.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.27.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.27.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.27.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.27.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.27.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.28.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.28.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.28.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.28.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.28.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.28.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.28.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.28.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.28.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.28.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.28.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.28.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.28.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.28.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.29.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.29.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.29.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.29.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.29.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.29.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.29.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.29.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.29.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.29.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.29.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.29.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.29.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.29.mlp.down_proj.lora_B.default'], 
    ['base_model.model.model.layers.30.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.30.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.30.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.30.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.30.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.30.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.30.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.30.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.30.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.30.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.30.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.30.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.30.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.30.mlp.down_proj.lora_B.default'], ['base_model.model.model.layers.31.self_attn.q_proj.lora_A.default', 'base_model.model.model.layers.31.self_attn.q_proj.lora_B.default', 'base_model.model.model.layers.31.self_attn.k_proj.lora_A.default', 'base_model.model.model.layers.31.self_attn.k_proj.lora_B.default', 'base_model.model.model.layers.31.self_attn.v_proj.lora_A.default', 'base_model.model.model.layers.31.self_attn.v_proj.lora_B.default', 'base_model.model.model.layers.31.self_attn.o_proj.lora_A.default', 'base_model.model.model.layers.31.self_attn.o_proj.lora_B.default', 'base_model.model.model.layers.31.mlp.gate_proj.lora_A.default', 'base_model.model.model.layers.31.mlp.gate_proj.lora_B.default', 'base_model.model.model.layers.31.mlp.up_proj.lora_A.default', 'base_model.model.model.layers.31.mlp.up_proj.lora_B.default', 'base_model.model.model.layers.31.mlp.down_proj.lora_A.default', 'base_model.model.model.layers.31.mlp.down_proj.lora_B.default']
    ]
}

REMIND_MODEL_BLOCK_DICT = {
    'resnet18': [['model_G.initial'],
                 ['model_G.group1.blocks.block0'],
                 ['model_G.group1.blocks.block1'],
                 ['model_G.group2.blocks.block0'],
                 ['model_G.group2.blocks.block1'],
                 ['model_G.group3.blocks.block0'],
                 ['model_G.group3.blocks.block1'],
                 ['model_F.group4.blocks.block0'],
                 ['model_F.group4.blocks.block1'],
                 ['model_F.pool', 'fc']],
    
    'resnet32': [['model_G.initial'],
                 ['model_G.group1.blocks.block0'],
                 ['model_G.group1.blocks.block1'],
                 ['model_G.group1.blocks.block2'],
                 ['model_G.group1.blocks.block3'],
                 ['model_G.group1.blocks.block4'],
                 ['model_G.group2.blocks.block0'],
                 ['model_G.group2.blocks.block1'],
                 ['model_G.group2.blocks.block2'],
                 ['model_G.group2.blocks.block3'],
                 ['model_G.group2.blocks.block4'],
                 ['model_G.group3.blocks.block0'],
                 ['model_G.group3.blocks.block1'],
                 ['model_G.group3.blocks.block2'],
                 ['model_F.group3.blocks.block3'],
                 ['model_F.group3.blocks.block4'],
                 ['model_F.pool','fc']],
    
    'vit': [['model_G.0'], 
            ['model_G.1'], 
            ['model_G.2'], 
            ['model_G.3'], 
            ['model_G.4'], 
            ['model_G.5'], 
            ['model_G.6'], 
            ['model_G.7'], 
            ['model_G.8'], 
            ['model_G.9'], 
            ['model_F.0'], 
            ['model_F.1'], 
            ['model_F.2'], 
            ['norm'],['fc']]
}

MEMO_MODEL_BLOCK_DICT = {
    'resnet18': [['backbone.initial'],
                 ['backbone.group1.blocks.block0'],
                 ['backbone.group1.blocks.block1'],
                 ['backbone.group2.blocks.block0'],
                 ['backbone.group2.blocks.block1'],
                 ['backbone.group3.blocks.block0'],
                 ['backbone.group3.blocks.block1'],
                 ['AdaptiveExtractors.0.group4.blocks.block0'],
                 ['AdaptiveExtractors.0.group4.blocks.block1'],
                 ['AdaptiveExtractors.0.pool'], ['fc']],
    
    'resnet32': [['backbone.initial'],
                 ['backbone.group1.blocks.block0'],
                 ['backbone.group1.blocks.block1'],
                 ['backbone.group1.blocks.block2'],
                 ['backbone.group1.blocks.block3'],
                 ['backbone.group1.blocks.block4'],
                 ['backbone.group2.blocks.block0'],
                 ['backbone.group2.blocks.block1'],
                 ['backbone.group2.blocks.block2'],
                 ['backbone.group2.blocks.block3'],
                 ['backbone.group2.blocks.block4'],
                 ['AdaptiveExtractors.0.group3.blocks.block0'],
                 ['AdaptiveExtractors.0.group3.blocks.block1'],
                 ['AdaptiveExtractors.0.group3.blocks.block2'],
                 ['AdaptiveExtractors.0.group3.blocks.block3'],
                 ['AdaptiveExtractors.0.group3.blocks.block4'],
                 ['AdaptiveExtractors.0.pool'], ['fc']],
    
    'vit': [['backbone.0'], 
            ['backbone.1'], 
            ['backbone.2'], 
            ['backbone.3'], 
            ['backbone.4'], 
            ['backbone.5'], 
            ['backbone.6'], 
            ['backbone.7'], 
            ['AdaptiveExtractors.0.0'], 
            ['AdaptiveExtractors.0.1'],
            ['AdaptiveExtractors.0.2'], 
            ['AdaptiveExtractors.0.3'], 
            ['AdaptiveExtractors.0.4'], 
            ['norm', 'fc']]
}

def get_llava_blockwise_flops(flops_dict, model_name="llava"):

    assert model_name in MODEL_BLOCK_DICT.keys()
    block_list = MODEL_BLOCK_DICT[model_name]
    
    forward_flops = []
    backward_flops = []
    
    for block in block_list:
        forward_flops.append(sum([flops_dict[layer]['forward_flops']/10e9 for layer in block]))
        backward_flops.append(sum([flops_dict[layer]['backward_flops']/10e9 for layer in block]))

    return forward_flops, backward_flops

def get_blockwise_flops(flops_dict, model_name, method=None):

    if method=="memo":
        assert model_name in MEMO_MODEL_BLOCK_DICT.keys()
        block_list = MEMO_MODEL_BLOCK_DICT[model_name]
    elif method=="remind":
        assert model_name in REMIND_MODEL_BLOCK_DICT.keys()
        block_list = REMIND_MODEL_BLOCK_DICT[model_name]
    else:
        assert model_name in MODEL_BLOCK_DICT.keys()
        block_list = MODEL_BLOCK_DICT[model_name]
    
    forward_flops = []
    backward_flops = []
    G_forward_flops = []
    G_backward_flops = []
    F_forward_flops = []
    F_backward_flops = []
    G_forward, G_backward, F_forward, F_backward = [], [], [], []
    
    for block in block_list:
        forward_flops.append(sum([flops_dict[layer]['forward_flops']/10e9 for layer in block]))
        backward_flops.append(sum([flops_dict[layer]['backward_flops']/10e9 for layer in block]))
    
        if method=="remind":
            for layer in block:
                if "model_G" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                else:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
                    
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))
            
        elif method=="memo":
            for layer in block:
                if "backbone" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                elif "AdaptiveExtractors" in layer or "fc" in layer:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
  
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))


    return forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops
