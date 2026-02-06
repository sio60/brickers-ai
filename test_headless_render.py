import asyncio
import os
from pathlib import Path
from route.headless_renderer import HeadlessPdfService

async def main():
    # Test LDR Content (Simple 2x4 Brick)
    ldr_content = """0 Name: Test Model
0 STEP
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
0 STEP
1 2 20 0 0 1 0 0 0 1 0 0 0 1 3001.dat
"""
    
    print("----------------------------------------------------------------")
    print("🧪 Headless Renderer Test")
    print("----------------------------------------------------------------")
    
    try:
        print(f"1. Calling capture_step_images...")
        step_images = await HeadlessPdfService.capture_step_images(ldr_content)
        
        print(f"2. Result: {len(step_images)} steps captured.")
        
        for i, views in enumerate(step_images):
            print(f"   - Step {i+1}: {len(views)} views captured.")
            for j, img_bytes in enumerate(views):
                fname = f"test_step{i+1}_view{j+1}.png"
                Path(fname).write_bytes(img_bytes)
                print(f"     -> Saved {fname} ({len(img_bytes)} bytes)")
                
        print("\n✅ Test Passed!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
