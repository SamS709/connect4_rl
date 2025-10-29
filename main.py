import os
from graphics.ai_models_interface import ai_models_interfaceApp

if __name__ == "__main__":
    os.environ['KIVY_GL_BACKEND'] = 'sdl2'  # Use pure SDL2 backend for better compatibility
    try:
        ai_models_interfaceApp().run()
    except Exception as e:
        import traceback
        print("Kivy app crashed with error:")
        traceback.print_exc()
        
 