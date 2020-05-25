from flask import Flask
from flask import request
from sys import argv, exit
from LSTM_API import LSTM_API
from datetime import datetime
from user_tokens import users
import pdb
#QuoteMachine = Model.QuoteComparison( Model.get_quotes( Model.load_data() ) )

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome to the LSTM api.\n Available models: '/models'\n Parameters: startdate,enddate,output_type,token,model_name."

@app.route('/models/')
@app.route('/models')
def linear():
    startdate = request.args.get('startdate',None)
    enddate = request.args.get('enddate',None)
    output_type = request.args.get('output_type')
    key = request.args.get('token',None)
    model_name = request.args.get('model_name',None)

    #Pretty hackable authentication request
    #-----------------------------
    print(users.values())
    print(key)
    if key not in users.values():
        return b"Access Denied"
    #--------------------------------

    #If no start date is defined return the list of available models to choose from
    #--------------------------------------------------------------------
    if model_name == None:
        available_models = ""
        for model in API:
            available_models = available_models + model.get_name()+"<br>"
        return "Choose from models: <br>" + available_models
    #--------------------------------------------------------------------

    #If model is specified, select from library of models, return the list if it is not found
    #----------------------------------------------------------------
    return_model = None
    models = []
    for model in API:
        if model_name == model.get_name():
            models.append( model.get_name() )
            return_model = model

    if return_model == None :
        return "Model " + model_name +" not found in " +str(models)
    #----------------------------------------------------------------

    #If the startdate is too early or missing notify the user
    #---------------------------------------------------------
    try:
        if datetime.strptime(startdate,'%Y-%m-%d') < return_model.lookup_table.index[0]:
            return "Earliest date available : " + str( return_model.lookup_table.index[0] )
    except:
        return "Include startdate=YYYY-MM-DD after " + str( return_model.lookup_table.index[0] )
    #-----------------------------------------------------------------------------------


    if output_type == "json":
    	return return_model.lookup_table[startdate:enddate].to_json()
    else :
    	return return_model.lookup_table[startdate:enddate].to_csv()

if __name__ == '__main__':
    if len(argv) < 3:
        print("wheat_api.py  <data.csv> <model_config1> <model_config2> ... ")
        exit(0)

    data_file = argv[1]
    API = []
    for i in range(2,len(argv)):
        # Api = LSTM_API(argv[1],argv[2])
        API.append( LSTM_API( argv[i], data_file ) )

    # pdb.set_trace()
    # app.run(host='0.0.0.0', port=63000,debug=True)
    app.run(debug=False)
