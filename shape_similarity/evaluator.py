# ============================================================================ 
# 3D 모델 형상 유사도 평가 모듈
# 이 파일은 원본 3D 모델(GLB)과 레고로 변환된 모델(LDR) 간의
# 형상 유사도를 정량적으로 평가하는 기능을 제공합니다.
# 두 모델을 복셀화하여 IoU(Intersection over Union) 점수를 계산하고,
# 차이점을 시각화하는 기능을 포함합니다.
# ============================================================================ 
import os
import sys
import numpy as np
import trimesh

# 상위 디렉토리를 경로에 추가하여 physical_verification 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physical_verification.ldr_loader import LdrLoader
from physical_verification.part_library import get_part_geometry

def load_ldr_as_mesh(ldr_path: str) -> trimesh.Trimesh:
    """
    LDR 파일을 파싱하여 모든 브릭을 포함하는 단일 trimesh.Trimesh 객체로 변환합니다.

    Args:
        ldr_path (str): LDR 파일 경로

    Returns:
        trimesh.Trimesh: 모든 브릭의 형상이 조합된 3D 메쉬
    """
    print(f"LDR 파일 로드 중: {ldr_path}")
    loader = LdrLoader()
    plan = loader.load_from_file(ldr_path)
    
    all_brick_meshes = []
    
    # LdrLoader가 파싱한 Brick 객체들은 ldr_loader 내부의 좌표계 변환을 거친 상태.
    # 정확한 형상 비교를 위해서는 LDraw 원본 좌표계의 파트 지오메트리를 가져와서
    # LDraw 변환 행렬을 적용해야 합니다.
    # (현재 ldr_loader는 AABB 기반이라, 여기서는 새로 파싱 로직을 간소화하여 사용)

    with open(ldr_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('1'):
                continue

            parts = line.split()
            try:
                # 1 <color> x y z a b c d e f g h i <file>
                pos = np.array([float(p) for p in parts[2:5]])
                rot_matrix = np.array([float(p) for p in parts[5:14]]).reshape((3, 3))
                part_id = " ".join(parts[14:])

                # 파트의 기본 지오메트리(삼각형) 가져오기
                geometry_tris = get_part_geometry(part_id)
                if not geometry_tris:
                    print(f"경고: '{part_id}'의 지오메트리를 찾을 수 없습니다. 건너뜁니다.")
                    continue

                vertices = [v for tri in geometry_tris for v in tri]
                faces = np.arange(len(vertices)).reshape(-1, 3)
                
                # trimesh 객체 생성
                brick_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                # LDR 변환 행렬 적용 (회전 + 이동)
                transform = np.eye(4)
                transform[:3, :3] = rot_matrix
                transform[:3, 3] = pos
                brick_mesh.apply_transform(transform)
                
                all_brick_meshes.append(brick_mesh)
            except Exception as e:
                print(f"LDR 라인 파싱 오류: {line} - {e}")

    if not all_brick_meshes:
        raise ValueError("LDR 파일에서 유효한 브릭을 찾을 수 없습니다.")

    # 모든 브릭 메쉬를 하나로 합침
    combined_mesh = trimesh.util.concatenate(all_brick_meshes)
    print("LDR 모델 메쉬 변환 완료.")
    return combined_mesh

def calculate_iou(voxel_grid1: trimesh.voxel.VoxelGrid, voxel_grid2: trimesh.voxel.VoxelGrid) -> float:
    """
    두 복셀 그리드의 IoU (Intersection over Union) 점수를 계산합니다.

    Args:
        voxel_grid1: 첫 번째 모델의 복셀 그리드
        voxel_grid2: 두 번째 모델의 복셀 그리드

    Returns:
        float: 0과 1 사이의 IoU 점수
    """
    matrix1 = voxel_grid1.matrix
    matrix2 = voxel_grid2.matrix
    
    intersection = np.sum(matrix1 & matrix2)
    union = np.sum(matrix1 | matrix2)
    
    if union == 0:
        return 1.0  # 두 모델이 모두 비어있으면 100% 동일하다고 간주
        
    return intersection / union

def visualize_difference(source_voxels, ldr_voxels, pitch, output_path="difference.glb"):
    """
    두 복셀 그리드의 차이점를 시각화하여 3D 모델로 저장합니다.

    Args:
        source_voxels: 원본 모델의 복셀 그리드
        ldr_voxels: LDR 모델의 복셀 그리드
        pitch (float): 복셀의 크기
        output_path (str): 저장할 파일 경로
    """
    source_matrix = source_voxels.matrix
    ldr_matrix = ldr_voxels.matrix

    missing_matrix = source_matrix & (~ldr_matrix)
    extra_matrix = ldr_matrix & (~source_matrix)

    missing_voxels = trimesh.voxel.VoxelGrid(missing_matrix, transform=source_voxels.transform)
    extra_voxels = trimesh.voxel.VoxelGrid(extra_matrix, transform=ldr_voxels.transform)

    missing_mesh = missing_voxels.as_boxes(colors=[255, 0, 0, 128]) # 빨간색: 누락된 부분
    extra_mesh = extra_voxels.as_boxes(colors=[0, 0, 255, 128])   # 파란색: 초과된 부분

    if missing_mesh.is_empty and extra_mesh.is_empty:
        print("시각적 차이점 없음.")
        return None

    # 원본 메시를 회색으로 함께 표시
    source_mesh = source_voxels.as_boxes(colors=[128, 128, 128, 64])

    # 모든 차이점 메쉬를 하나로 합침
    scene = trimesh.Scene([source_mesh, missing_mesh, extra_mesh])
    
    print(f"차이점 시각화 모델을 '{output_path}'에 저장합니다.")
    scene.export(output_path)
    return output_path


def compare_models(glb_path: str, ldr_path: str, voxel_pitch: float = None, resolution: int = 128):
    """
    GLB와 LDR 모델의 형상 유사도를 평가합니다.

    Args:
        glb_path (str): 원본 GLB 파일 경로
        ldr_path (str): 비교할 LDR 파일 경로
        voxel_pitch (float, optional): 복셀 크기. 지정하지 않으면 해상도 기준으로 자동 계산.
        resolution (int, optional): 복셀화 해상도. voxel_pitch가 없을 때 사용.

    Returns:
        float: 유사도 점수 (0-100)
    """
    try:
        # 1. 모델 로드
        print("="*50)
        source_mesh = trimesh.load(glb_path, force='mesh')
        ldr_mesh = load_ldr_as_mesh(ldr_path)
        print("="*50)
        
        # 2. 모델 정렬 (ICP)
        print("모델 정렬 시작 (ICP)...")
        # 원점을 질량 중심으로 이동
        source_mesh.apply_translation(-source_mesh.center_mass)
        ldr_mesh.apply_translation(-ldr_mesh.center_mass)

        # 크기 정규화 (가장 긴 축 기준)
        source_mesh.apply_scale(1.0 / source_mesh.extents.max())
        ldr_mesh.apply_scale(1.0 / ldr_mesh.extents.max())
        
        # ICP를 위한 포인트 클라우드 샘플링
        source_points = source_mesh.sample(2048)
        ldr_points = ldr_mesh.sample(2048)
        
        # ICP 실행하여 변환 행렬 찾기
        icp_transform, _, _ = trimesh.registration.icp(ldr_points, source_points)
        
        # LDR 메쉬에 변환 적용하여 정렬
        ldr_mesh.apply_transform(icp_transform)
        print("모델 정렬 완료.")
        print("="*50)

        # 3. 복셀화 (수정된 로직)
        print("복셀화 진행...")
        # 두 메쉬를 모두 포함하는 공통 공간(Scene) 생성
        scene = trimesh.Scene([source_mesh, ldr_mesh])

        if voxel_pitch is None:
            pitch = scene.extents.max() / resolution
        else:
            pitch = voxel_pitch

        # 공통 공간의 원점과 형태를 계산
        grid_origin = scene.bounds[0]
        grid_shape = np.ceil(scene.extents / pitch).astype(int)

        # 동일한 그리드 설정으로 각 메쉬를 개별적으로 복셀화
        source_voxels = trimesh.voxel.creation.voxelize(source_mesh, pitch, origin=grid_origin, shape=grid_shape)
        ldr_voxels = trimesh.voxel.creation.voxelize(ldr_mesh, pitch, origin=grid_origin, shape=grid_shape)

        print(f"복셀화 완료 (Pitch: {pitch:.4f}, Shape: {source_voxels.shape}).")
        print("="*50)

        # 4. 유사도 계산
        print("유사도 점수 계산 (IoU)...")
        iou_score = calculate_iou(source_voxels, ldr_voxels)
        similarity_percent = iou_score * 100
        print(f"유사도 점수: {similarity_percent:.2f}%")
        print("="*50)

        # 5. 차이점 시각화
        visualize_difference(source_voxels, ldr_voxels, pitch)
        
        return similarity_percent

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == '__main__':
    print("형상 유사도 평가 스크립트 실행")
    
    # --- 테스트 설정 ---
    # 실제 파일 경로로 변경해야 합니다.
    # 예시: GLB 파일과 LDR 파일을 스크립트와 같은 폴더에 위치시킨 경우
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # !!! 중요 !!!
    # 아래 파일명은 실제 존재하는 파일로 변경해야 합니다.
    # glb_file = "model.glb"
    # ldr_file = "model.ldr"
    
    # 테스트용 임시 파일 생성 (실제 실행 시에는 주석 처리)
    # ---------------------------------------------------
    # 가짜 GLB (큐브)
    temp_glb_path = os.path.join(current_dir, "temp_cube.glb")
    cube_mesh = trimesh.creation.box(extents=[1, 1, 1])
    cube_mesh.export(temp_glb_path)
    
    # 가짜 LDR (큐브와 유사하지만 약간 다름)
    temp_ldr_path = os.path.join(current_dir, "temp_cube.ldr")
    with open(temp_ldr_path, "w") as f:
        # 1x1x1 브릭 8개로 2x2x2 큐브 만들기
        f.write("0 A 2x2x2 cube made of 8 1x1 bricks\n")
        positions = [
            [-10, -12, -10], [10, -12, -10], [-10, -12, 10], [10, -12, 10],
            [-10, 12, -10], [10, 12, -10], [-10, 12, 10], [10, 12, 10],
        ]
        for i, pos in enumerate(positions):
            f.write(f"1 15 {pos[0]} {pos[1]} {pos[2]} 1 0 0 0 1 0 0 0 1 3005.dat\n")
        # 중앙에 구멍을 만들기 위해 하나 제거
        # f.write("1 15 10 12 10 1 0 0 0 1 0 0 0 1 3005.dat\n") 
    # ---------------------------------------------------

    # --- 실행 ---
    # compare_models 함수에 실제 파일 경로 전달
    # score = compare_models(glb_path=os.path.join(current_dir, glb_file), 
    #                        ldr_path=os.path.join(current_dir, ldr_file))
    
    score = compare_models(glb_path=temp_glb_path, ldr_path=temp_ldr_path)
    
    if score > 0:
        print(f"\n최종 유사도 점수: {score:.2f} / 100")

    # 임시 파일 삭제
    os.remove(temp_glb_path)
    os.remove(temp_ldr_path)
