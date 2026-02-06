import os
import asyncio
import base64
from pathlib import Path
from typing import List
from playwright.async_api import async_playwright

# Project Root determination (consistent with kids_render.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = PROJECT_ROOT / "public"
RENDERER_HTML_PATH = PUBLIC_DIR / "renderer.html"

class HeadlessPdfService:
    @staticmethod
    async def capture_step_images(ldr_content: str) -> List[List[bytes]]:
        """
        LDR 콘텐츠를 받아 Headless Browser에서 렌더링 후
        각 스텝별 3개 뷰(View)의 이미지를 캡처하여 반환합니다.
        
        Returns:
            List[List[bytes]]: [Step1_Images, Step2_Images, ...]
            각 Step_Images는 3개의 bytes(PNG) 리스트
        """
        
        # HTML 파일 존재 확인
        if not RENDERER_HTML_PATH.exists():
            raise FileNotFoundError(f"Renderer HTML not found at: {RENDERER_HTML_PATH}")
            
        file_url = f"file://{RENDERER_HTML_PATH}"
        
        step_images_list = []
        
        async with async_playwright() as p:
            # 브라우저 실행 (Headless)
            # args: GPU 가속 활성화 (WebGL 성능 향상 시도, 일부 환경에선 불안정할 수 있음)
            # --use-gl=egl 등은 환경따라 다름. 일단 기본값.
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--use-gl=angle", "--use-angle=gl"] 
            )
            
            # Context 생성 (Viewport 설정해두면 좋음)
            context = await browser.new_context(viewport={"width": 1024, "height": 768})
            page = await context.new_page()
            
            print(f"[Headless] Loading renderer: {file_url}")
            await page.goto(file_url)
            
            # 페이지 로드 대기 (Three.js 초기화 등)
            # window.loadLdrContent 함수가 정의될 때까지 대기
            await page.wait_for_function("typeof window.loadLdrContent === 'function'")
            
            # 1. LDR 로드
            # JS 함수 호출: window.loadLdrContent(ldr_content) -> returns totalSteps
            # LDR 텍스트 전송 시 이스케이프 주의 (evaluate 인자로 전달하면 playwright가 처리해줌)
            total_steps = await page.evaluate("text => window.loadLdrContent(text)", ldr_content)
            print(f"[Headless] Loaded LDR. Total Steps: {total_steps}")
            
            if total_steps == 0:
                print("[Headless] No steps found or parse error.")
                await browser.close()
                return []
                
            # 2. 각 스텝 순회
            for step_idx in range(total_steps):
                # 스텝 렌더링
                await page.evaluate("idx => window.renderStep(idx)", step_idx)
                
                # 렌더링 안정화 대기 (100ms)
                await asyncio.sleep(0.2) 
                
                current_step_blobs = []
                
                # 3개 뷰 캡처
                for view_idx in range(3):
                    # 카메라 이동
                    await page.evaluate("v => window.setCameraView(v)", view_idx)
                    
                    # 카메라 이동 및 렌더링 대기
                    await asyncio.sleep(0.1)
                    
                    # 스크린샷 캡처
                    # canvas 요소만 캡처할 수도 있고, 전체 화면 캡처할 수도 있음.
                    # renderer.html 스타일이 body margin 0, canvas full screen 이므로 page.screenshot() 무방
                    screenshot_bytes = await page.screenshot(type="png")
                    current_step_blobs.append(screenshot_bytes)
                
                step_images_list.append(current_step_blobs)
                
                if (step_idx + 1) % 5 == 0:
                    print(f"[Headless] Captured step {step_idx + 1}/{total_steps}")
            
            await browser.close()
            
        return step_images_list
