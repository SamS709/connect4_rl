import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
from kivy.app import App
from kivy.uix.label import Label

class TestApp(App):
    def build(self):
        return Label(text='Hello Kivy!')

if __name__ == '__main__':
    TestApp().run()