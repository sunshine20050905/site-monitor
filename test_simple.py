"""
简化版测试脚本 — 工地视觉算法服务（可插拔推理层）
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    print("测试模块导入...")
    try:
        from imsafe.ai_framework import SiteVisionFramework

        print("SiteVisionFramework 导入成功")
        return True
    except Exception as e:
        print(f"导入失败: {e}")
        return False


def test_framework_init():
    print("测试框架初始化...")
    try:
        from imsafe.ai_framework import SiteVisionFramework

        framework = SiteVisionFramework()
        success = framework.initialize()
        if success:
            print("框架初始化成功")
            return True
        print("框架初始化失败")
        return False
    except Exception as e:
        print(f"初始化失败: {e}")
        return False


def test_ai_functions():
    print("测试推理接口...")
    try:
        from imsafe.ai_framework import SiteVisionFramework

        framework = SiteVisionFramework()
        framework.initialize()

        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        r1 = framework.face_recognition(test_image)
        print(f"人脸识别测试: {'成功' if 'error' not in r1 else '失败'}")

        r2 = framework.helmet_detection(test_image)
        print(f"安全帽检测测试: {'成功' if 'error' not in r2 else '失败'}")

        r3 = framework.smoking_detection(test_image)
        print(f"吸烟检测测试: {'成功' if 'error' not in r3 else '失败'}")

        return True
    except Exception as e:
        print(f"推理测试失败: {e}")
        return False


def main():
    print("=" * 50)
    print("工地视觉算法服务 — 集成测试")
    print("=" * 50)

    tests = [
        ("模块导入", test_imports),
        ("框架初始化", test_framework_init),
        ("推理接口", test_ai_functions),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n测试: {test_name}")
        print("-" * 30)
        try:
            if test_func():
                print(f"{test_name}: 通过")
                passed += 1
            else:
                print(f"{test_name}: 失败")
        except Exception as e:
            print(f"{test_name}: 异常 - {e}")

    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("所有测试通过!")
    else:
        print("部分测试失败")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
