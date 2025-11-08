import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphics.ai_models_interface import InfoLabel
from kivy.app import App
from kivy.properties import ListProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.lang import Builder
import matplotlib
from global_vars import *
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image as PILImage

# Load the .kv file for MyButton styling
Builder.load_file(os.path.join(os.path.dirname(__file__), 'model_info.kv'))

try:
    from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("kivy.garden.matplotlib not installed. Install with: pip install kivy_garden.matplotlib")
    MATPLOTLIB_AVAILABLE = False

from scripts.logger import Logger
import numpy as np

class MyDropDown(Button): # A rounded button with borders that appears multiple times in the interface
    button_color = ListProperty(LIGHT_BLUE)
    line_button_color = ListProperty(WHITE)
    line_width = NumericProperty(2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_color = LIGHT_BLUE[:]
        self.pressed_color = DARK_BLUE[:]
        
    def on_press(self):
        """Change color when button is pressed"""
        self.button_color = self.pressed_color
        
    def on_release(self):
        """Restore original color when button is released"""
        self.button_color = self.original_color
    
    

class ModelInfoWindow(BoxLayout):
    
    def __init__(self, model_name, n_model, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.model_name = model_name
        self.n_model = n_model
        
        # Set background color to LIGHT_BLUE
        from kivy.graphics import Color, Rectangle
        with self.canvas.before:
            Color(*LIGHT_BLUE)
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_bg, pos=self._update_bg)
        
        # Load data from logger
        self.logger = Logger(model_name=model_name, n_model=n_model)
        self.load_data()
        
        # Create UI
        self.create_ui()
    
    def _update_bg(self, *args):
        """Update background rectangle when size/position changes"""
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos
    
    def load_data(self):
        """Load training history from logger"""
        try:
            self.total_epochs, self.evaluation_epochs, self.win_rates, \
            self.forbidden_rates, self.lose_rates = self.logger.get_current_infos()
            
            # Determine available depths
            if len(self.win_rates) > 0 and len(self.win_rates[0]) > 0:
                self.n_depths = len(self.win_rates[0])
                self.available_depths = list(range(self.n_depths))
            else:
                self.n_depths = 0
                self.available_depths = []
            
            self.current_depth = 0 if self.n_depths > 0 else None
        except Exception as e:
            print(f"Error loading data: {e}")
            self.total_epochs = 0
            self.evaluation_epochs = []
            self.win_rates = []
            self.forbidden_rates = []
            self.lose_rates = []
            self.n_depths = 0
            self.available_depths = []
            self.current_depth = None
    
    def create_ui(self):
        """Create the user interface"""
        # Top bar with model info and depth selector
        top_bar = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=10)
        
        # Model info label
        info_text = f"Model: {self.model_name}{self.n_model} | Total Epochs: {self.total_epochs} | Evaluations: {len(self.evaluation_epochs)}"
        self.info_label = InfoLabel(text=info_text, size_hint_x=0.7)
        top_bar.add_widget(self.info_label)
        
        # Depth selector dropdown
        if self.n_depths > 0:
            self.depth_button = MyDropDown(text=f'Depth: {self.current_depth}     V', size_hint_x=0.3, size_hint_y=None, height=35, pos_hint={'top': 1})
            self.depth_dropdown = DropDown(height = 1)
            
            for depth in self.available_depths:
                btn = MyDropDown(text=f'Depth {depth}', size_hint_y=None, size_hint_x=None, width=150, height=35)
                # Manually apply MyButton styling (since dropdown doesn't use .kv styling)
                btn.button_color = btn.button_color  # Triggers the property
                btn.line_button_color = btn.line_button_color
                btn.bind(on_release=lambda btn, d=depth: self.select_depth(d))
                self.depth_dropdown.add_widget(btn)
            
            self.depth_button.bind(on_release=self.depth_dropdown.open)
            
            top_bar.add_widget(self.depth_button)
        else:
            no_data_label = Label(text='No data available', size_hint_x=0.3)
            top_bar.add_widget(no_data_label)
        
        self.add_widget(top_bar)
        
        # Plot area
        self.plot_container = BoxLayout(orientation='vertical', size_hint_y=0.9, padding=[10, 10, 10, 10])
        self.add_widget(self.plot_container)
        
        # Initial plot
        if self.current_depth is not None:
            self.update_plot()
    
    def select_depth(self, depth):
        """Handle depth selection from dropdown"""
        self.current_depth = depth
        self.depth_button.text = f'Depth: {depth}     V'
        self.depth_dropdown.dismiss()
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with data for the selected depth"""
        print(f"Updating plot for depth: {self.current_depth}")
        self.plot_container.clear_widgets()
        
        if self.current_depth is None or len(self.win_rates) == 0:
            no_data_label = Label(text='No evaluation data available')
            self.plot_container.add_widget(no_data_label)
            return
        
        # Extract data for current depth
        win_rate_depth = [wr[self.current_depth] for wr in self.win_rates]
        forbidden_rate_depth = [fr[self.current_depth] for fr in self.forbidden_rates]
        lose_rate_depth = [lr[self.current_depth] for lr in self.lose_rates]
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.set_title(f'{self.model_name}{self.n_model} - Performance vs Minimax (Depth {self.current_depth})', fontsize=14)
        ax.set_xlabel('Training Epochs', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plot the three metrics
        ax.plot(self.evaluation_epochs, win_rate_depth, marker='o', label='Win Rate', 
                linewidth=2, color='green', markersize=8)
        ax.plot(self.evaluation_epochs, forbidden_rate_depth, marker='s', label='Forbidden Move Rate', 
                linewidth=2, color='orange', markersize=8)
        ax.plot(self.evaluation_epochs, lose_rate_depth, marker='^', label='Loss Rate', 
                linewidth=2, color='red', markersize=8)
        
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1])
        
        # Remove whitespace around the plot
        plt.tight_layout(pad=0.1)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.06)
        
        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        pil_img = PILImage.open(buf)
        
        # Add black border to the image
        from PIL import ImageDraw
        border_width = 2
        draw = ImageDraw.Draw(pil_img)
        width, height = pil_img.size
        # Draw rectangle border
        for i in range(border_width):
            draw.rectangle(
                [(i, i), (width - 1 - i, height - 1 - i)],
                outline='black'
            )
        
        # Use unique filename to force Kivy to reload the image
        import time
        temp_path = f'/tmp/model_plot_{self.current_depth}_{int(time.time() * 1000)}.png'
        pil_img.save(temp_path)
        
        # Add plot to Kivy
        img = Image(source=temp_path, allow_stretch=True)
        img.reload()  # Force reload
        self.plot_container.add_widget(img)
        
        buf.close()
        plt.close(fig)


class ModelInfoApp(App):
    
    def __init__(self, model_name, n_model, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.n_model = n_model
    
    def build(self):
        return ModelInfoWindow(model_name=self.model_name, n_model=self.n_model)


if __name__ == '__main__':
    # Test with a model
    import sys
    
    if len(sys.argv) > 2:
        model_name = sys.argv[1]
        n_model = sys.argv[2]
    else:
        model_name = "expert"
        n_model = "1"
    
    ModelInfoApp(model_name=model_name, n_model=n_model).run()
