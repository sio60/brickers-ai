"""
LDR Converter - 파츠 DB 로드 + JSON 파싱

BrickParts_Database.json 로드, JSON→BrickModel 변환
"""

import json
from typing import Dict

from .models import Vector3, PlacedBrick, BrickModel


def load_parts_db(filepath: str) -> Dict:
    """BrickParts_Database.json 로드 후 partId로 인덱싱"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # partId를 키로 하는 딕셔너리 생성
    parts_dict = {}
    for part in data['parts']:
        parts_dict[part['partId']] = part

    return parts_dict


def parse_brick_model(json_data: dict) -> BrickModel:
    """JSON 딕셔너리를 BrickModel로 변환"""
    bricks = []
    for b in json_data.get('bricks', []):
        pos = b['position']
        brick = PlacedBrick(
            id=b['id'],
            part_id=b['partId'],
            position=Vector3(x=pos['x'], y=pos['y'], z=pos['z']),
            rotation=b['rotation'],
            color_code=b['colorCode'],
            layer=b['layer']
        )
        bricks.append(brick)

    return BrickModel(
        model_id=json_data['modelId'],
        name=json_data['name'],
        mode=json_data['mode'],
        bricks=bricks,
        target_age=json_data.get('targetAge'),
        created_at=json_data.get('createdAt')
    )
