from vjepa2.models_v2_1.utils.modules import Block as VJEPAEncoderBlock
from vjepa2.models.utils.modules import Block as VJEPAPoolerBlock, CrossAttentionBlock as VJEPAPoolerCrossAttentionBlock
from model.modules import BasicTransformerBlock


def get_fsdp_shard_modules():
    shard_modules = {
        BasicTransformerBlock,
        VJEPAEncoderBlock,
        VJEPAPoolerBlock,
        VJEPAPoolerCrossAttentionBlock,
    }
    return shard_modules


def get_fsdp_checkpointing_modules():
    checkpointing_modules = {
        BasicTransformerBlock,
        VJEPAEncoderBlock,
        VJEPAPoolerBlock,
        VJEPAPoolerCrossAttentionBlock,
    }
    return checkpointing_modules
