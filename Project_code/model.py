import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2,drop_prob = 0.75):
        super(DecoderRNN, self).__init__()
        self.output_size = vocab_size
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
            

    
    def forward(self, features, captions):
        
        embeds = self.embedding(captions[:,:-1])
       
        x = torch.cat((features.unsqueeze(1), embeds), 1)
     
      
      
        x,_ = self.lstm(x)
       
       # x = x.view(x.size()[0]*x.size()[1], self.output_size)
        
        ## TODO: put x through the fully-connected layer
        x = self.fc(x)
       # print("final size ",x.size())

        # return x and the hidden state (h, c)
        return x
      


    def sample(self, inputs, states=None, max_len=15):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
       
        out = []
        
       # print("initial ",inputs.size())
        
        for i in range(max_len):
            output,states = self.lstm(inputs,states)   
          #  print("After LSTM ",output.size())
            x  = self.fc(output)
           # print("After fc ",x.size())
            #return x
            prediction = x.argmax(dim=2)
            out.append(prediction[0].item())
            inputs = self.embedding(prediction)
            
            ##print(prediction.size())
            #print("After embedding ",inputs.size())
        return out
    
    

                        
           
        
        
        pass