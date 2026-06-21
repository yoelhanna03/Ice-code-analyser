import torch
from transformers import AutoTokenizer
from ice.main import IceModule  # Importe ton module Lightning

def predict_vulnerability(code_snippet, checkpoint_path):
    # 1. Charger le tokenizer de CodeBERT
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # 2. Recharger le modèle à partir du meilleur checkpoint sauvegardé
    # (Remplace les hparams par ceux utilisés ou laisse par défaut)
    class Args:
        num_classes = 2
        freeze_encoder = False
        lr = 2e-5
    
    model = IceModule.load_from_checkpoint(checkpoint_path, hparams_arg=Args())
    model.eval()  # Passer en mode évaluation (désactive le Dropout et le BatchNorm)
    
    # 3. Préparer le code fourni (Tokenization)
    inputs = tokenizer(
        code_snippet,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # 4. Envoyer le code à l'IA sans calculer les gradients (plus rapide)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
    
    # 5. Interpréter le résultat
    classes = {0: "Sûr (Safe)", 1: "Vulnérable (Vulnerable)"}
    print("\n" + "="*40)
    print("🔮 RÉSULTAT DE L'ANALYSE :")
    print(f"Statut : {classes[prediction]}")
    print(f"Confiance : {probabilities[0][prediction].item() * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Remplace par le vrai nom de ton fichier de checkpoint
    CHECKPOINT = "./checkpoints/TON_FICHIER_ICI.ckpt" 
    
    # Pose ta question (colle le code à analyser ici)
    CODE_A_TESTER = """
    void buffer_overflow_example(char *str) {
        char buffer[16];
        strcpy(buffer, str); 
    }
    """
    
    predict_vulnerability(CODE_A_TESTER, CHECKPOINT)