import json
import os
from datetime import datetime

class Logger:
    
    def __init__(self, model_name, n_model):
        self.model_name = model_name
        self.n_model = n_model  # Store n_model for use in save()
        # Get project root (connect_4_dqn directory) - one level up from scripts/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.registration_path = os.path.join(project_root, "models", model_name)
        print(self.registration_path)
        self.file = self.load_file(n_model)
        self.save()
        
    def load_file(self, n_model):
        history_file = os.path.join(self.registration_path, f"training_history{n_model}.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            history = {"total_epochs": 0,"losses": {"loss": [], "epoch": []} , "algorithm": "", "evaluations": []}
        return history
    
    def overwrite_epochs(self, epochs):
        self.file["total_epochs"] = epochs
        self.save()
        
    def write_loss(self,loss, epoch):
        self.file["losses"]["epoch"].append(epoch)
        self.file["losses"]["loss"].append(loss)
        self.save()
        
    def set_algo_name(self, name):
        self.file["algorithm"] = name
        self.save()
        
    def add_new_evaluation(self, n_games, winrate, forbidden_rate, lose_rate):
        evaluation = {
            "epoch": self.file["total_epochs"]
        }
        for i in range(len(winrate)):
            evaluation[f"depth_{i}"] = {
                "n_games": n_games,
                "win_rate": winrate[i],
                "forbidden_rate": forbidden_rate[i],
                "lose_rate": lose_rate[i]
            }
        self.file["evaluations"].append(evaluation)
        self.save()
        
    def get_current_epochs(self):
        return self.file["total_epochs"]
    
    def get_model_algo(self):
        return self.file["algorithm"]
        
    def get_current_infos(self):
        """
        Returns training history information
        
        Returns:
            total_epochs (int): Total number of training epochs
            evaluation_epochs (list): List of epoch numbers when evaluations were performed
            win_rates (list[list]): Win rates for each evaluation at each depth
            forbidden_rates (list[list]): Forbidden move rates for each evaluation at each depth
            lose_rates (list[list]): Loss rates for each evaluation at each depth
        """
        total_epochs = self.file["total_epochs"]
        evaluation_epochs = []
        win_rates = []
        forbidden_rates = []
        lose_rates = []
        
        for evaluation in self.file["evaluations"]:
            evaluation_epochs.append(evaluation["epoch"])
            
            # Extract metrics for all depths in this evaluation
            win_rate_depths = []
            forbidden_rate_depths = []
            lose_rate_depths = []
            
            # Find all depth keys (depth_0, depth_1, etc.)
            depth_keys = sorted([k for k in evaluation.keys() if k.startswith("depth_")])
            
            for depth_key in depth_keys:
                depth_data = evaluation[depth_key]
                win_rate_depths.append(depth_data["win_rate"])
                forbidden_rate_depths.append(depth_data["forbidden_rate"])
                lose_rate_depths.append(depth_data["lose_rate"])
            
            win_rates.append(win_rate_depths)
            forbidden_rates.append(forbidden_rate_depths)
            lose_rates.append(lose_rate_depths)
        
        return total_epochs, evaluation_epochs, win_rates, forbidden_rates, lose_rates
    
    def save(self):
        """Save the current state of self.file to disk"""
        history_file = os.path.join(self.registration_path, f"training_history{self.n_model}.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "w") as f:
            json.dump(self.file, f, indent=2)
    

if __name__ == "__main__":
    logger = Logger("logger_test2")
    logger.overwrite_epochs(10)
    logger.add_new_evaluation(100,[0.1,0.2,0.3],[0.2,0.1,0.1],[0.7,0.7,0.6])

    