# Reference:
# https://github.com/Pierre-Sassoulas/pySankey.git
# https://www.kaggle.com/code/kooaslansefat/cnn-97-accuracy-plus-safety-monitoring-safeml/notebook#SafeML:-Safety-Monitoring-through-Statistical-Distance-Measure

from pysankey import sankey
import matplotlib.pyplot as plt
import pandas as pd

classes = { 0:'0: Speed limit (20km/h)',
        1:'1: Speed limit (30km/h)', 
        2:'2: Speed limit (50km/h)', 
        3:'3: Speed limit (60km/h)', 
        4:'4: Speed limit (70km/h)', 
        5:'5: Speed limit (80km/h)', 
        6:'6: End of speed limit (80km/h)', 
        7:'7: Speed limit (100km/h)', 
        8:'8: Speed limit (120km/h)', 
        9:'9: No passing', 
        10:'10: No passing veh over 3.5 tons', 
        11:'11: Right-of-way at intersection', 
        12:'12: Priority road', 
        13:'13: Yield', 
        14:'14: Stop', 
        15:'15: No vehicles', 
        16:'16: Veh > 3.5 tons prohibited', 
        17:'17: No entry', 
        18:'18: General caution', 
        19:'19: Dangerous curve left', 
        20:'20: Dangerous curve right', 
        21:'21: Double curve', 
        22:'22: Bumpy road', 
        23:'23: Slippery road', 
        24:'24: Road narrows on the right', 
        25:'25: Road work', 
        26:'26: Traffic signals', 
        27:'27: Pedestrians', 
        28:'28: Children crossing', 
        29:'29: Bicycles crossing', 
        30:'30: Beware of ice/snow',
        31:'31: Wild animals crossing', 
        32:'32: End speed + passing limits', 
        33:'33: Turn right ahead', 
        34:'34: Turn left ahead', 
        35:'35: Ahead only', 
        36:'36: Go straight or right', 
        37:'37: Go straight or left', 
        38:'38: Keep right', 
        39:'39: Keep left', 
        40:'40: Roundabout mandatory', 
        41:'41: End of no passing', 
        42:'42: End no passing veh > 3.5 tons' }

def create_sankey_plot(y_pred, y_true, model_name):
    Predictions_df = pd.DataFrame()
    Predictions_df['True'] = y_true
    Predictions_df['Pred'] = y_pred

    Predictions_df['True'] = Predictions_df['True'].map(classes)
    Predictions_df['Pred'] = Predictions_df['Pred'].map(classes)
    Predictions_df.head()

    # specify class order
    class_order = [classes[i] for i in range(len(classes))]

    # use sankey plot and specify left and right order
    sankey(
        left  = Predictions_df['True'], 
        right = Predictions_df['Pred'], 
        leftLabels=class_order,
        rightLabels=class_order,
        aspect=20, fontsize=10
    )

    # Get current figure
    fig = plt.gcf()

    # Set size in inches
    fig.set_size_inches(12, 24)

    # Set the color of the background to white
    fig.set_facecolor("#2b2d42")

    #font color to white
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.titlecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['font.family'] = 'Trebuchet MS'
    plt.rcParams['font.size'] = 30

    #Title
    plt.title("{} Predictions".format(model_name), fontsize=40, fontname='Trebuchet MS', pad=30)
    #subtitle 
    plt.suptitle("True Labels vs Predicted Labels", fontsize=20, fontname='Trebuchet MS', y=0.889)