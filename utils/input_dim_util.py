from config.configs import Config, FrameType

def get_input_dim(frame_type: FrameType, cfg: Config) -> int:
    """FrameType에 따라 필요한 input dimension을 계산하여 반환합니다."""
    if frame_type in [FrameType.RAW_RAW, FrameType.RAW_LOGITS]:
        return cfg.raw_input_dim
    elif frame_type in [FrameType.EMBEDDING_RAW, FrameType.EMBEDDING_LOGITS]:
        return cfg.embedding_input_dim
    elif frame_type in [FrameType.EMBEDDING_MIX_RAW, FrameType.EMBEDDING_MIX_LOGITS]:
        return cfg.mix_input_dim
    else:
        raise ValueError(f"Unknown FrameType for input dimension: {frame_type}")
