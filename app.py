from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertModel
import datetime

app = Flask(__name__)

log_file = 'log.txt'

# Load the state dictionary of the model
state_dict = torch.load('GoEmotions_model.pt', map_location=torch.device('cpu'))

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',  return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 14)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


# Create an instance of the model
model = BERTClass()
model.load_state_dict(state_dict)
model.to(torch.device('cpu'))

# Create an instance of the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add the enumerate function to Jinja2 environment
app.jinja_env.globals.update(enumerate=enumerate)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=175,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Load the trained model
    model.eval()

    # Make a prediction
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(torch.device('cpu'), dtype=torch.long)
        attention_mask = encoding['attention_mask'].to(torch.device('cpu'), dtype=torch.long)
        token_type_ids = encoding['token_type_ids'].to(torch.device('cpu'), dtype=torch.long)
        outputs = model(input_ids, attention_mask, token_type_ids)
        predictions = torch.sigmoid(outputs).cpu().detach().numpy().tolist()

    
    emotions = ['admiration' , 'anger' , 'approval' , 'disappointment' , 'confusion' , 'curiosity' , 'sadness' , 
                 'pride' , 'excitement' , 'gratitude' , 'surprise' , 'desire' , 'fear' , 'neutral']
    
    # Create a dictionary to store the input text and predictions
    rounded_predictions = [round(p, 2) for p in predictions[0]]
    prediction_dict = {'text': text, 'predictions': rounded_predictions, 'timestamp': str(datetime.datetime.now())}

    # Write the dictionary to the log file
    if prediction_dict['text']:
        with open(log_file, 'a') as f:
            f.write(str(prediction_dict) + '\n')

    return render_template('index.html', text=text, predictions=predictions,emotions=emotions)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)
