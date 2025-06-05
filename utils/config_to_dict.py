from config.configs import Config
from config.configs import DataType, ModelType, RunType, StackType, FrameType

def config_to_dict(config_obj: Config) -> dict:
    """Config 객체를 딕셔너리로 변환하여 JSON 직렬화가 가능하도록 합니다."""
    config_dict = {}
    for attr_name in dir(config_obj):
        # 비공개 속성, 호출 가능한 속성, 특정 내장 속성 제외
        if not attr_name.startswith('__') and not callable(getattr(config_obj, attr_name)):
            value = getattr(config_obj, attr_name)
            # Enum 타입은 .value로 변환
            if isinstance(value, (DataType, ModelType, RunType, StackType, FrameType)):
                config_dict[attr_name] = value.value
            else:
                config_dict[attr_name] = value
    return config_dict
