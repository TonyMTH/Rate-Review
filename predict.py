# Import libraries
from torchtext.vocab import FastText
import preprocess as pr
from parameters import *


class RateReview:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, title, body):
        # Load Model
        model = torch.load(self.model_path)

        # Preprocess Data
        sequence = pr.padding(pr.encoder(pr.preprocessing(str(title) + " " + str(body)), FastText("simple")),
                              max_seq_len=max_seq_len)
        sequence = torch.Tensor([sequence])
        sequence.resize_(sequence.size()[0], max_seq_len * emb_dim)

        # Predict
        model.eval()
        with torch.no_grad():
            output = model(sequence)
            _, pred = output.max(1)

        return pred.item()+1


if __name__ == '__main__':
    model_pth = 'data/best_model.pt'
    title = "Works, but not as advertised"
    #"2",
    body = """
    "Works, but not as advertised","I bought one of these chargers..the instructions say the lights stay on while 
    the battery charges...true. The instructions doNT say the lights turn off when its done. Which is also true. 24
     hours of charging and the lights stay on. I returned it thinking I had a bad unit.The new one did the same thing. 
     I just kept it since it does charge...but the lights are useless since they seem to always stay on. 
     It's a ""backup"" charger for when I manage to drain all my AAs but I wouldn't want this as my only charger."
           """
    out = RateReview(model_pth).predict(title, body)
    print(out)
