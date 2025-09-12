import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_vars import *
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Color, Rectangle, Line
from kivy.clock import mainthread
if __name__!="__main__":
    try:
        from global_vars import var1
    except:
        var1 = None

import shutil
import time
import threading
import keras

from kivy.properties import ListProperty, NumericProperty, StringProperty

class ThemedProgressBar(ProgressBar):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        

class GetInfo: # Retrieve the informations from a model
     
     def __init__(self,game="Connect4"):
          self.path = f"\{game}\AI\models"
     
     def get_model_path(self,model_name):
         return os.path.join(os.getcwd()+self.path,model_name)
     
     def get_model_names(self): # returns a list of the name of the models we have
          return os.listdir(os.getcwd()+self.path)
     
     def get_info_model(self,model_name): # returns the name, number of dense layers, number of neurons per dense layer
        print(os.getcwd())
        filepath=os.getcwd()+self.path+f"\{str(model_name)}1"
        print(filepath)
        model = keras.models.load_model(filepath=filepath)
        cfg = model.get_config()
        layers = cfg['layers']
        n_layers = len(layers)
        n_dense = 0
        n_neurons_per_layer = 0
        for i in range(n_layers):
            if layers[i]['class_name'] == "Dense":
                if cfg['layers'][i]['config']['units']>1:
                    n_neurons_per_layer = cfg['layers'][i]['config']['units']
                    n_dense += 1
        return model_name, n_dense, n_neurons_per_layer


class InfoLabel(Label): # The label with a border that appears multiple times in the interface
     
     line_width = NumericProperty(2)
     background_color = ListProperty(BLUE)
     line_button_color = ListProperty(WHITE)
     
     def __init__(self, **kwargs):
          super().__init__(**kwargs)
        

class MyButton(Button): # A rounded button with borders that appears multiple times in the interface
    button_color = ListProperty(LIGHT_BLUE)
    line_button_color = ListProperty(WHITE)
    line_width = NumericProperty(2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.getInfo = GetInfo()



class ScrollableLabel(ScrollView): # A Scrollable label used to display logs

    font_size1 = NumericProperty(20)
    text = StringProperty('')
    color = ListProperty(RED)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.font_size1 = 0.2*self.width


class ScrollingMenu(BoxLayout):

    # kivy properties you find in the graphics.kv file (TextArea class)
    text = StringProperty("Liste des modeles") # text displayed in the blue area
    title = StringProperty("Selectionne le modele d'IA de Ned") # title displayed in the brown area
    image_source = StringProperty('images/transparent.png') # source of the image displayed in the text area
    image_height = NumericProperty(30) # height of the image displayed
    TitleArea_color = ListProperty(SAND) # color of the title area
    Background_color = ListProperty(LIGHT_BLUE) # color of the background of the text area
    TA_color = ListProperty(BLUE) # color of the background of the text area
    Text_Color = ListProperty(WHITE) # color of the text
    Line_color = ListProperty(WHITE) # color of the line

    def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
    def press_refresh(self,instance):
        pass

    def release_refresh(self,instance):
        pass


class ScrollableBoxes(BoxLayout):

    line_width = NumericProperty(2)
    background_color = ListProperty(BLUE)
    line_button_color = ListProperty(WHITE)

    def __init__(self,D_models={}, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.D_models = D_models
        self.scroll = ScrollView(size_hint=(1,1))
        self.layout = BoxLayout(padding = [5,10,5,10], orientation="vertical",spacing=5, size_hint_y=None )
        self.layout.bind(minimum_height=self.layout.setter('height'))
        box=BoxLayout(size_hint_y=None, height=40)
        lbl = Label(font_name = "fonts/pixel.ttf",size_hint_y=None, height=40, text = "Modeles en cours d'entrainement:",valign="middle", halign="center", color=WHITE)
        lbl.bind(
                    size=lambda instance, value: (
                        setattr(instance, 'text_size', value),
                        setattr(instance, 'font_size', 0.04 * value[0])  # value[0] is width
                    )
                )
        box.add_widget(lbl)
        self.add_widget(box)
        self.scroll.add_widget(self.layout)
        self.add_widget(self.scroll)
        



class ChooseAIModel(BoxLayout): # Menu to choose the model to play against

    Background_color = ListProperty(LIGHT_BLUE)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        


    def on_kv_post(self, base_widget):
        self.getInfo = GetInfo(self.game)
        self.scroll_menu = ScrollingMenu()
        self.info_box = self.ids.info_box
        self.scroll_menu.press_refresh = self.press_refresh
        self.scroll_menu.release_refresh = self.release_refresh
        self.scroll = self.scroll_menu.ids.scroll 
        self.info_label = self.ids.info_label
        self.scroll_box = self.ids.Scroll_box
        self.text1 = "Nom du modele"
        self.text2 = "Nombre de couches"
        self.text3 = "Nombre de neurones par couche"
        self.info_label.text = f"{self.text1}: \n\n\n{self.text2}: \n\n\n{self.text3}:"
        self.init_buttons()
        self.setup_title()


    def press_refresh(self,instance):
        instance.size_hint = (None,0.4)

    def release_refresh(self,instance):
        instance.size_hint = (None,0.5)
        self.load_button_list()

        
    
    def init_buttons(self):
        self.layout = GridLayout(cols=1, spacing=10, size_hint_y=None )
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.L_buttons = []
        self.load_button_list()
        self.validate_button = self.ids.validate_button
        self.validate_button.bind(on_press = self.on_press_validate, on_release = self.on_release_validate)
        self.validate_button.button_color = LIGHT_GREEN

    @mainthread
    def load_button_list(self):
        self.layout.clear_widgets()
        self.scroll.clear_widgets()
        self.scroll_box.clear_widgets()
        self.model_list = self.getInfo.get_model_names()
        self.L_buttons = []
        for i in range(len(self.model_list)):
            btn = MyButton(text=str(self.model_list[i][:-1]), size_hint_y=None, height=40, on_press = self.on_press, on_release = self.on_release)
            if i % 2 == 0:
                self.L_buttons.append(btn)
                self.layout.add_widget(btn)
        self.scroll.add_widget(self.layout)
        self.scroll_box.add_widget(self.scroll_menu)

    def on_press_validate(self,instance):
        if self.validate_button.button_color == GREEN:
            self.validate_button.button_color = DARK_GREEN
    
    def on_release_validate(self,instance):
        if self.validate_button.button_color == DARK_GREEN:
            self.validate_button.button_color = LIGHT_GREEN
            var1.model_name = self.model_name
            print(var1)
            App.get_running_app().manager.push('Connect4Game')
            for btn in self.L_buttons:
                btn.button_color = LIGHT_BLUE

    def setup_title(self):
        self.ids.title_label.background_color = WHITE
        self.ids.title_label.color = BLACK
        self.ids.title_label.line_button_color = BLACK
         

    def on_release(self,instance):
        instance.button_color = BLUE
        try:
            thread1 = threading.Thread(target=self.get_info, args=(instance, ))
            thread1.start()
            self.validate_button.button_color = GREEN
        except BaseException:
            print('Error: unable to start thread')
    

    def on_press(self,instance):
        global MODEL_NAME
        for btn in self.L_buttons:
             btn.button_color = LIGHT_BLUE
        instance.button_color = DARK_BLUE
        self.model_name = instance.text
        MODEL_NAME = instance.text
        print(MODEL_NAME)
        

    def get_info(self,instance):
        try:
            model_name,  n_layers, n_neurons= self.getInfo.get_info_model(instance.text)
            info = f"{self.text1}: {model_name}\n\n\n{self.text2}: {n_layers}\n\n\n{self.text3}: {n_neurons}"
        except:
             info = "Erreur de chargement des informations"
        self.info_label.text = info

class EditModels(ChooseAIModel): # Menu to edit (create and delete) models
    def __init__(self, **kwargs):
          super().__init__(**kwargs)

    

    def on_kv_post(self, base_widget):
        super().on_kv_post(base_widget)
        self.log_label = ScrollableLabel()
        self.getInfo = GetInfo(self.game)
        self.model_name = ""
        self.bottom_box =  self.ids.bottom_box
        self.menu_train = MenuTrain(self.on_release_cancel_new, self.info_label,self.log_label,self.bottom_box,self.info_box)
        self.info_input = MenuInput(
    self.on_release_cancel_new,
    self.info_label,
    self.log_label,
    bottom_box=self.bottom_box,
    on_release_validate=self.on_release_validate_new, 
)
        self.update_texts_and_buttons()
    
    def on_release_validate_new(self,instance):
        if instance.button_color == DARK_GREEN:
            instance.button_color = GREEN
            self.scroll_box.clear_widgets()
            self.scroll_box.add_widget(self.scroll_menu)            
            self.model_name = self.info_input.children[1].children[2].children[0].children[1].text
            try:
                n_layers = int(self.info_input.children[1].children[1].children[0].children[1].text)
            except:
                n_layers = 0
            try:
                n_neurons_per_layer = int(self.info_input.children[1].children[0].children[0].children[1].text)
            except:
                n_neurons_per_layer = 0
            n_neurons_tot = n_layers*n_neurons_per_layer
            print(f"Model name: {self.model_name}, Number of layers: {n_layers}, Neurons per layer: {n_neurons_per_layer}, Total neurons: {n_neurons_tot}")
            self.log_label.text = ""
            if self.model_name == "":
                self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                self.log_label.text+="\n[ERREUR] Le modele n'a pas pu etre créé \nDonne un nom valide à ton modele"
            if n_neurons_per_layer == 0:
                self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                self.log_label.text+="\n[ERREUR] Le modele n'a pas pu etre créé \nLe modele ne peut pas avoir 0 neurone !"
            if n_layers == 0:
                self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                self.log_label.text+="\n[ERREUR] Le modele n'a pas pu etre créé \nLe modele ne peut pas avoir 0 couche de neurones !"
            if n_neurons_per_layer > 512:
                self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                self.log_label.text+="\n[ERREUR] Le modele n'a pas pu etre créé \nLe modele ne peut pas avoir plus de 512 neurones par couche !"
            if n_layers > 10:
                self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                self.log_label.text+="\n[ERREUR] Le modele n'a pas pu etre créé \nLe modele ne peut pas avoir plus de 10 couches !"
            else:
                t1 = threading.Thread(target=self.create_model, args=(n_layers, n_neurons_per_layer))
                t1.start()
            self.info_input.children[1].children[2].children[0].children[1].text = ""
            self.info_input.children[1].children[1].children[0].children[1].text = ""
            self.info_input.children[1].children[0].children[0].children[1].text = ""

    def create_model(self,n_layers, n_neurons_per_layer):
        self.model_list = self.getInfo.get_model_names()
        print(self.model_list)
        for model in self.model_list:
            if self.model_name.upper() == model[:-1].upper():
                self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                self.log_label.text = "\n[ERREUR] Le modele n'a pas pu etre créé \nCe nom est deja pris !"
        if self.log_label.text == "":
            self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
            try:
                DQN(reset=False,model_name=self.model_name,n_neurons=n_neurons_per_layer,n_layers=n_layers,softmax_=False,P1="1")
                DQN(reset=False,model_name=self.model_name,n_neurons=n_neurons_per_layer,n_layers=n_layers,softmax_=False,P1="2")
                self.log_label.text = f"\n[color=3EB64B][INFO] Le modele {self.model_name} a été créé avec succès.[/color]"
            except:
                self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                self.log_label.text = "\n[ERREUR] Le modele n'a pas pu etre créé \nErreur inconnue !"
            self.load_button_list()

    def on_release_cancel_new(self):
        self.scroll_box.clear_widgets()
        self.scroll_box.add_widget(self.scroll_menu)
        self.info_input.children[1].children[2].children[0].children[1].text = ""
        self.info_input.children[1].children[1].children[0].children[1].text = ""
        self.info_input.children[1].children[0].children[0].children[1].text = ""
        
    
    def update_texts_and_buttons(self):
        self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
        self.bottom_box.size_hint = [1,0.1]
        self.bottom_box.clear_widgets()
        self.bottom_box.add_widget(self.log_label)
        self.left_title = self.scroll_menu.ids.left_title
        self.left_title.text = "Liste des modeles"
        self.actions = BoxLayout(size_hint=(1,0.3),orientation="horizontal",padding = [0.1*self.width,0,0.1*self.width,0.1*self.height], spacing = 10)
        self.actions2 = BoxLayout(size_hint=(1,0.3),orientation="horizontal",padding = [0.1*self.width,0,0.1*self.width,0.1*self.height], spacing = 10)

        self.add = MyButton(text="New", on_press=self.add_on_press, on_release = self.add_on_release)
        self.delete = MyButton(text="Delete", on_press=self.delete_on_press, on_release = self.delete_on_release)
        self.train = MyButton(text="Train", on_press=self.train_on_press, on_release = self.train_on_release)
        self.add.button_color=GREEN
        self.delete.button_color=LIGHT_RED
        self.train.button_color=LIGHT_BLUE
        self.actions.add_widget(self.delete)
        self.actions.add_widget(self.train)
        self.actions2.add_widget(self.add)
        self.scroll_menu.add_widget(self.actions)
        self.scroll_menu.add_widget(self.actions2)


    def log_error(self,i):
        if i == 1:
            self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
            self.log_label.text+"\n[ERREUR] Impossible de supprimer le modele "+str(self.model_name) + self.log_label.text
        if i == 2:
            self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
            self.log_label.text="\n[ERREUR] Impossible de supprimer les modeles presents par defaut" + self.log_label.text


    def on_press(self, instance):
        global MODEL_NAME
        for btn in self.L_buttons:
             btn.button_color = LIGHT_BLUE
        instance.button_color = DARK_BLUE
        self.model_name = instance.text
        MODEL_NAME = instance.text
        print(MODEL_NAME)
        self.train.button_color = BLUE
        self.delete.button_color = RED
    
    def train_on_press(self,instance):
        if instance.button_color == BLUE:
            instance.button_color = DARK_BLUE

    def train_on_release(self,instance):
        if instance.button_color == DARK_BLUE:
            instance.button_color = BLUE
            self.scroll_box.clear_widgets()
            self.info_label.text = f"Nombre d'epoques: \n\n\nTaux d'apprentissage: \n\n\nFacteur de reduction:"
            #self.scroll_box.padding = [0,0,0,0]
            self.scroll_box.add_widget(self.menu_train)
        
    
    def delete_on_release(self,instance):
        instance.button_color = LIGHT_RED      

    def delete_on_press(self,instance):
        if instance.button_color==RED:
            instance.button_color=DARK_RED
            path = self.getInfo.get_model_path(self.model_name)
            print(path)
            print(self.model_name)
            cond = self.model_name == "Expert" or self.model_name == "Intermediaire" or self.model_name == "Debutant"
            if cond:
                self.log_error(2)
            else:
                try:
                    shutil.rmtree(path+"1")
                    self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
                    self.log_label.text = f"\n[color=3EB64B][INFO] Le modèle {self.model_name} a été supprimé avec succès.[/color]"
                except:
                    self.log_error(1)
                try:
                    shutil.rmtree(path+"2")
                except:
                    pass
                self.load_button_list()

    def add_on_press(self,instance):
        if instance.button_color == GREEN:
            instance.button_color = DARK_GREEN

    def add_on_release(self,instance):
        if instance.button_color == DARK_GREEN:
            instance.button_color = GREEN
            self.scroll_box.clear_widgets()
            self.info_label.text = f"Nom du modele: \n\n\nNombre de couches: \n\n\nNombre de neurones par couche:"
            #self.scroll_box.padding = [0,0,0,0]
            self.scroll_box.add_widget(self.info_input)

class IBeamTextInput(TextInput):
    _instances = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        IBeamTextInput._instances.append(self)
        if len(IBeamTextInput._instances) == 1:
            Window.bind(mouse_pos=IBeamTextInput.on_mouse_pos_all)

    def on_parent(self, *args):
        if self.parent is None and self in IBeamTextInput._instances:
            IBeamTextInput._instances.remove(self)
            if not IBeamTextInput._instances:
                Window.unbind(mouse_pos=IBeamTextInput.on_mouse_pos_all)

    @staticmethod
    def on_mouse_pos_all(window, pos):
        for instance in IBeamTextInput._instances:
            if instance.get_root_window() and instance.collide_point(*instance.to_widget(*pos)):
                Window.set_system_cursor('ibeam')
                return
        Window.set_system_cursor('arrow')

class InfoInput(BoxLayout):

    button_set_color = ListProperty(GREEN)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def press(self):
        print(1)
    
    def release(self):
        pass

class MenuInput(BoxLayout):

    validate_color = ListProperty(GREEN)
    cancel_color = ListProperty(RED)
    button_set_color0 = ListProperty(GREEN)
    button_set_color1 = ListProperty(GREEN)
    button_set_color2 = ListProperty(GREEN)

    text_input0 = StringProperty("")
    text_input1 = StringProperty("")
    text_input2 = StringProperty("")



    def __init__(self, on_release_cancel,info_label,log_label,bottom_box,on_release_validate=None, **kwargs):
        self.on_release_cancel_ = on_release_cancel
        if on_release_validate is not None:
            self.on_release_validate = on_release_validate
        self.info_label = info_label
        self.log_label = log_label
        self.bottom_box = bottom_box
        super().__init__(**kwargs)

    def on_kv_post(self, base_widget):
        super().on_kv_post(base_widget)
        self.text1 = "Nom du modele"
        self.text2 = "Nombre de couches"
        self.text3 = "Nombre de neurones par couche"
        self.info_label.text = f"{self.text1}: \n\n\n{self.text2}: \n\n\n{self.text3}:"
    
    def log_info(self,i):
        self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
        if i== 0:
            self.log_label.text="\n[ERREUR] Le nom du modele ne peut pas etre vide."+self.log_label.text
        if i == 1:
            self.log_label.text="\n[ERREUR] Le nombre de neurones doit être compris entre 1 et 512."+self.log_label.text
        if i == 2:
            self.log_label.text="\n[ERREUR] Le nombre de couches doit être compris entre 1 et 10."+self.log_label.text

    def on_press_cancel(self,instance):
        if instance.button_color == RED:
            instance.button_color = DARK_RED
        print("COUCOU")

    def set_on_press0(self):
        if self.children[1].children[2].children[0].children[1].text == "":
            self.log_info(0)
            self.button_set_color0 = LIGHT_RED
        else:
            self.button_set_color0 = DARK_GREEN
            self.info_label.text = f"{self.text1}: {self.children[1].children[2].children[0].children[1].text} \n\n\n{self.text2}: {self.children[1].children[1].children[0].children[1].text} \n\n\n{self.text3}: {self.children[1].children[0].children[0].children[1].text}"

    def set_on_press1(self):
        if self.children[1].children[1].children[0].children[1].text == "" or float(self.children[1].children[1].children[0].children[1].text) < 1 or float(self.children[1].children[1].children[0].children[1].text) > 10:
            self.log_info(1)
            self.button_set_color1 = LIGHT_RED
        else:
            self.button_set_color1 = DARK_GREEN
            self.info_label.text = f"{self.text1}: {self.children[1].children[2].children[0].children[1].text} \n\n\n{self.text2}: {self.children[1].children[1].children[0].children[1].text} \n\n\n{self.text3}: {self.children[1].children[0].children[0].children[1].text} "


    def set_on_press2(self):
        if self.children[1].children[0].children[0].children[1].text == "" or float(self.children[1].children[0].children[0].children[1].text) < 1 or float(self.children[1].children[0].children[0].children[1].text) > 512:
            self.log_info(2)
            self.button_set_color2 = LIGHT_RED
        else:
            self.button_set_color2 = DARK_GREEN
            self.info_label.text = f"{self.text1}: {self.children[1].children[2].children[0].children[1].text} \n\n\n{self.text2}: {self.children[1].children[1].children[0].children[1].text} \n\n\n{self.text3}: {self.children[1].children[0].children[0].children[1].text} "
    
    def set_on_release0(self):
        if self.button_set_color0 == LIGHT_RED:
            self.button_set_color0 = LIGHT_GREEN
        else:
            self.button_set_color0 = GREEN

    def set_on_release1(self):
        if self.button_set_color1 == LIGHT_RED:
            self.button_set_color1 = LIGHT_GREEN
        else:
            self.button_set_color1 = GREEN

    def set_on_release2(self):
        if self.button_set_color2 == LIGHT_RED:
            self.button_set_color2 = LIGHT_GREEN
        else:
            self.button_set_color2 = GREEN

        

    
    def on_release_cancel(self,instance):
        if instance.button_color == DARK_RED:
            instance.button_color = RED
        self.on_release_cancel_()

    def on_press_validate(self,instance):
        if instance.button_color == GREEN:
            instance.button_color = DARK_GREEN

class InfoBoxLayout(BoxLayout):
    line_width = NumericProperty(2)
    background_color = ListProperty(BLUE)
    line_button_color = ListProperty(WHITE)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
class MenuTrain(MenuInput):

    def __init__(self,on_release_cancel,info_label,log_label,bottom_box,info_box, **kwargs):
        super().__init__(on_release_cancel,info_label,log_label,bottom_box,**kwargs)
        self.button_set_color0 = LIGHT_GREEN
        self.button_set_color1 = LIGHT_GREEN
        self.button_set_color2 = LIGHT_GREEN
        self.info_box = info_box
        infoboxlayout = InfoBoxLayout(padding = [4,4,4,4])
        infoboxlayout.add_widget(self.scrollable_label)
        self.info_box.add_widget(infoboxlayout)
        self.L_training = [] # dict of the models currently training {model_name: n_epochs}
    
    def on_kv_post(self, base_widget):
        global MODEL_NAME
        self.children[1].children[2].children[0].children[1].text = ""
        self.children[1].children[1].children[0].children[1].text = ""
        self.children[1].children[0].children[0].children[1].text = ""
        self.scrollable_label = ScrollableBoxes()
        self.scrollable_label.font_size1 = 0.1*self.width
        super().on_kv_post(base_widget)
        self.text1 = "Nombre d'epoques"
        self.text2 = "Taux d'apprentissage"
        self.text3 = "Facteur de reduction"
        self.filter1 = "int"
        self.filter2 = "float"
        self.titre = "Entraine le modele" + MODEL_NAME
        self.green_text = "Entrainer"
        self.epochs = None
        self.learning_rate = None
        self.discount_factor = None
        self.validate_color = LIGHT_GREEN
        self.info_label.text = f"{self.text1}: \n\n\n{self.text2}: \n\n\n{self.text3}:"

    @mainthread
    def log_info(self,i,model_name = ""):
        self.bottom_box.padding = [0,0.05*self.bottom_box.height,0,0.05*self.bottom_box.height]
        if i == 0:
            self.log_label.text = "\n[ERREUR] Le nombre d'epoques doit être compris entre 1 et 10000."+self.log_label.text
        if i == 1:
            self.log_label.text = "\n[ERREUR] Le taux d'apprentissage doit être compris entre 0.00001 et 0.1"+self.log_label.text
        if i == 2:
            self.log_label.text = "\n[ERREUR] Le facteur de réduction doit être compris entre 0.1 et 0.999"+self.log_label.text
        if i == 3:
            self.log_label.text = "\n[ERREUR] Tu dois remplir tous les champs avec une valeur valide avant de lancer l'entrainement."+self.log_label.text
        if i == 4:
            self.log_label.text = "\n[ERREUR] Ce modèle est déjà en cours d'entrainement."+self.log_label.text
        if i == 5:
            self.log_label.text = "\n[color=3EB64B][INFO] L'entrainement du modèle "+model_name+" a été lancé avec succès.[/color]"+self.log_label.text
        if i == 6:
            self.log_label.text = "\n[color=3EB64B][INFO] L'entrainement du modèle "+model_name+" est terminé.[/color]"+self.log_label.text
        if i == 7:
            self.log_label.text = "\n[ERREUR] Tu ne peux pas entrainer plus de 3 modeles a la fois."+self.log_label.text

    
    def set_on_press0(self):
        if self.children[1].children[2].children[0].children[1].text == "" or int(self.children[1].children[2].children[0].children[1].text) < 1 or int(self.children[1].children[2].children[0].children[1].text) > 10000:
            self.log_info(0)
            self.button_set_color0 = LIGHT_RED
            self.epochs = None
        else:
            if self.button_set_color1 == GREEN and self.button_set_color2 == GREEN:
                self.validate_color = GREEN
            self.epochs = int(self.children[1].children[2].children[0].children[1].text)
            self.info_label.text = f"{self.text1}: {self.children[1].children[2].children[0].children[1].text} \n\n\n{self.text2}: {self.children[1].children[1].children[0].children[1].text} \n\n\n{self.text3}: {self.children[1].children[0].children[0].children[1].text} "

    def set_on_press1(self):
        if self.children[1].children[1].children[0].children[1].text == "" or float(self.children[1].children[1].children[0].children[1].text) < 0.00001 or float(self.children[1].children[1].children[0].children[1].text) > 0.1:
            self.log_info(1)
            self.button_set_color1 = LIGHT_RED
            self.learning_rate = None
        else:
            if self.button_set_color0 == GREEN and self.button_set_color2== GREEN:
                self.validate_color = GREEN
            self.learning_rate = float(self.children[1].children[1].children[0].children[1].text)
            self.info_label.text = f"{self.text1}: {self.children[1].children[2].children[0].children[1].text} \n\n\n{self.text2}: {self.children[1].children[1].children[0].children[1].text} \n\n\n{self.text3}: {self.children[1].children[0].children[0].children[1].text} "

    def set_on_press2(self):
        if self.children[1].children[0].children[0].children[1].text == "" or float(self.children[1].children[0].children[0].children[1].text) < 0.1 or float(self.children[1].children[0].children[0].children[1].text) > 0.999:
            self.log_info(2)
            self.button_set_color2 = LIGHT_RED
            self.discount_factor = None
        else:
            if self.button_set_color1 == GREEN and self.button_set_color0 == GREEN:
                self.validate_color = GREEN
            self.discount_factor = float(self.children[1].children[0].children[0].children[1].text)
            self.info_label.text = f"{self.text1}: {self.children[1].children[2].children[0].children[1].text} \n\n\n{self.text2}: {self.children[1].children[1].children[0].children[1].text} \n\n\n{self.text3}: {self.children[1].children[0].children[0].children[1].text} "

    def on_press_validate(self,instance):
        if instance.button_color == LIGHT_GREEN:
            if self.children[1].children[2].children[0].children[1].text == "" or int(self.children[1].children[2].children[0].children[1].text) < 1 or int(self.children[1].children[2].children[0].children[1].text) > 10000:
                pass
            else:
                self.epochs = int(self.children[1].children[2].children[0].children[1].text)
            if self.children[1].children[1].children[0].children[1].text == "" or float(self.children[1].children[1].children[0].children[1].text) < 0.00001 or float(self.children[1].children[1].children[0].children[1].text) > 0.1:
                pass
            else:
                self.learning_rate = float(self.children[1].children[1].children[0].children[1].text)
            if self.children[1].children[0].children[0].children[1].text == "" or float(self.children[1].children[0].children[0].children[1].text) < 0.1 or float(self.children[1].children[0].children[0].children[1].text) > 0.999:    
                pass
            else:    
                self.discount_factor = float(self.children[1].children[0].children[0].children[1].text)
            instance.button_color = DARK_GREEN
        if instance.button_color == GREEN:
            instance.button_color = DARK_GREEN
    
    def on_release_validate(self,instance):
        if instance.button_color == DARK_GREEN:
            if self.epochs is not None and self.learning_rate is not None and self.discount_factor is not None and MODEL_NAME not in self.L_training and len(self.L_training) < 3:
                lbl = Label(font_name = "fonts/pixel.TTF",halign= 'left',size_hint_y=0.7, text = "Nom du medele: "+str(MODEL_NAME) + "\nNombre d'epoques: 0 / "+str(self.epochs))
                lbl.bind(
                            size=lambda instance, value: (
                                setattr(instance, 'text_size', value),
                                setattr(instance, 'font_size', 0.04 * value[0])  # value[0] is width
                            )
                        )
                pb = ThemedProgressBar(size_hint_y=0.3, max=self.epochs, value=0)
                box = InfoBoxLayout(orientation = "vertical",size_hint_y=None, height=70,spacing=10, padding=[10,0,10,10])
                box.add_widget(lbl)
                box.add_widget(pb)
                self.L_training.append(MODEL_NAME)
                self.scrollable_label.layout.add_widget(box)                
                t = threading.Thread(target=self.train, args=(self.epochs, self.learning_rate, self.discount_factor,lbl,box,pb))
                t.start()
                self.log_info(5, MODEL_NAME)
            elif MODEL_NAME in self.L_training:
                self.log_info(4)     
            elif len(self.L_training) >= 3:
                self.log_info(7)
            else:
                self.log_info(3)
            instance.button_color = LIGHT_GREEN

    def on_release_cancel(self, instance):
        self.button_set_color0 = LIGHT_GREEN
        self.button_set_color1 = LIGHT_GREEN    
        self.button_set_color2 = LIGHT_GREEN
        self.validate_color = LIGHT_GREEN
        super().on_release_cancel(instance)

    def train(self, epochs, learning_rate, discount_factor,lbl,box,pb):
        global MODEL_NAME
        if True:
            print(MODEL_NAME)
            print(lbl.text)
            model_name = MODEL_NAME
            trainer = Train(model_name=MODEL_NAME,learning_rate=learning_rate,discount_factor=discount_factor,info_label=lbl,scrollable_lablel=self.scrollable_label,box = box,pb = pb)
            trainer.P1vsP2(epochs)
            self.log_info(6, model_name)
            self.L_training.remove(model_name)
    
class BoxLayoutWithLine(BoxLayout):
    line_width = NumericProperty(2)
    background_color = ListProperty(BLUE)
    line_button_color = ListProperty(WHITE)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ai_models_interfaceApp(App):
      def build(self):
            return EditModels()
      

if __name__!="__main__":
    Builder.load_file(os.path.join(os.path.dirname(__file__), 'ai_models_interface.kv'))

if __name__=="__main__":
    var1 = 1
    #model = keras.models.load_model(filepath="C:\Dev\ned_project\Connect4\AI\models\my_linear_model1")
    ai_models_interfaceApp().run()
    """train = Train(reset=False, model_name="ELO", softmax_=False, eps=0.5)
    train.P1vsP2(100)"""

