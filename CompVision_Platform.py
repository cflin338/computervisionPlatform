import numpy as np
import tensorflow as tf
import cv2 as cv
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table
import plotly.graph_objs as go
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
   

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(children = [
    html.H1("Feature/Object Detection Platform"),
    #html.Br(),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    
    html.Br(),
    html.Img(id = 'img_placement'),
    html.Br(),
    html.Div(children = [html.Div(dcc.Dropdown(options = [{'label':'Feature Detection', 'value':'Feature Detection'},
                                                          {'label':'Object Detection', 'value':'Object Detection'}],
                                               id = 'Feature_Object_Dropdown',
                                               placeholder = 'Select Detection Type',), 
                                  className = 'five columns'),
                         
                         html.Div(dcc.Dropdown(id = 'CV_Dropdown_Options',), 
                                  className = 'five columns'),
                         html.Div(html.Button('Show Params', id = 'param-button', n_clicks = 0),
                                  className = 'two columns')],
             id = 'dropdown-grouping',
             className = 'row'),
        
    html.Br(),
    html.Div(children = [html.Button('Display', id = 'Display_Detection', n_clicks = 0, 
                                     className = 'one columns',style={"width": "250px", 
                                                                      "height": "50px"})], 
             id = 'params-buttons-row', className = 'row'),
    html.Br(), 
    html.Div(children = [], id = 'altered-img-plot-holder'),
    html.Br(),
    html.Footer('end'),
    
])
    
@app.callback(Output(component_id = 'img_placement', component_property = 'src'),
              Input('upload-image', 'contents'))
def show_img(list_of_contents):
    if list_of_contents is not None:
        return list_of_contents[0]

@app.callback([Output(component_id = 'CV_Dropdown_Options', component_property = 'options')],
              [Input(component_id = 'Feature_Object_Dropdown', component_property = 'value')])
def select_cv_mode(dropdown_items):
    if dropdown_items is None:
        return [{'label':'tmp', 'value': 'Detection Options Displayed Here'}]
    radio_list = []
    if dropdown_items == 'Feature Detection':
        
        radio_list = [{'label':'HarrisCorner', 'value': 'HarrisCorner'}, #implemented
                      {'label': 'SIFT', 'value': 'SIFT'}, #implemented
                      {'label': 'SURF', 'value': 'SURF'}, #implemented, not working
                      {'label': 'FAST', 'value': 'FAST'}, #implemented
                      {'label': 'BRIEF', 'value':'BRIEF'}, #not implemented
                      {'label':'ORB', 'value': 'ORB'}] #not implemented
    elif dropdown_items=='Object Detection':
        radio_list = [{'label': 'YOLO', 'value': 'YOLO'},
                      {'label': 'R-CNN', 'value':'R-CNN'}]
    return [radio_list]

@app.callback([Output(component_id = 'params-buttons-row', component_property = 'children')], 
              [Input(component_id = 'param-button', component_property = 'n_clicks')],
               [State(component_id = 'Feature_Object_Dropdown', component_property = 'value'),
               State(component_id = 'CV_Dropdown_Options', component_property = 'value')])

def display_param_options(n_clicks, Feature_Object, selection):
    children = []
    if Feature_Object == 'Feature Detection':
        if selection == 'HarrisCorner':
            children.append(html.H6('BlockSize',className = 'one columns'))
            children.append(dcc.Input(id = 'blocksize', value = 2, type = 'number', 
                                           className = 'one columns', placeholder = 'blocksize',
                                           ))
            children.append(html.H6('ksize', className = 'one columns'))
            children.append(dcc.Input(id = 'ksize', value = 3, type = 'number', 
                                           className = 'one columns', placeholder = 'ksize',
                                           ))
            children.append(html.H6('k', className = 'one columns'))
            children.append(dcc.Input(id = 'k', value = .04, type = 'number', 
                                           className = 'one columns', placeholder = 'k',
                                           ))
            children.append(html.H6('threshold', className = 'one columns'))
            children.append(dcc.Input(id = 'harris-threshold', value = .01, type = 'number', 
                                           className = 'one columns', placeholder = 'threshold',
                                           ))
        elif selection == 'SIFT':
            children.append(html.H6('No parameters', className = 'two columns'))
        elif selection == 'SURF':
            children.append(html.H6('threshold', className = 'one columns'))
            children.append(dcc.Input(id = 'surf-threshold', value = 5000, type = 'number', 
                                           className = 'one columns', placeholder = 'threshold',
                                           ))
            children.append(html.H6('Upright', className = 'one columns'))
            children.append(dcc.Dropdown(options = [{'label':'True', 'value': True},
                                                                    {'label':'False', 'value': False}], 
                                         id = 'surf-Upright',className = 'one columns', placeholder = 'Upright',
                                         ))
        elif selection == 'FAST':
            children.append(html.H6('threshold', className = 'one columns'))
            children.append(dcc.Input(id = 'fast-threshold', value = 5000, type = 'number', 
                                           className = 'one columns', placeholder = 'threshold',
                                           ))
            children.append(html.H6('NM-Suppression', className = 'one columns'))
            children.append(dcc.Input(id = 'fast-nmSuppression', options = [{'label':'True', 'value': True},
                                                                        {'label':'False', 'value': False}], 
                                           className = 'one columns', placeholder = 'threshold',
                                           ))
            
    children.append(html.Button('Display', id = 'Display_Detection', n_clicks = 0, 
                                className = 'one columns',style={"width": "250px", "height": "50px"}))
    return [children]

@app.callback([Output(component_id = 'altered-img-plot-holder', component_property = 'children')], 
              [Input(component_id = 'Display_Detection', component_property = 'n_clicks')],
              [State(component_id = 'params-buttons-row', component_property = 'children'),
               State(component_id = 'Feature_Object_Dropdown', component_property = 'value'),
               State(component_id = 'CV_Dropdown_Options', component_property = 'value'),
               State(component_id = 'upload-image', component_property = 'contents')
               ])

def display_CV_(n_clicks, params, FeatureObjectChoice, cv_option, img_url):
    if img_url is not None:
        img = cv.imread(img_url[0])
        
        children = [html.Div('dummy')]
    
        if n_clicks>0:
            param_list = {}
            for p in params:
                if p['type']=='Input':
                    param_list[p['props']['id']] = p['props']['value']
            
            children.append(html.Div('dummy2'))  
        
        if cv_option=='HarrisCorner':
            #print('harris selected')
            fig=px.imshow(img)
            c = dcc.Graph(figure = fig)
            return [c]
            #parameters:
            #   blocksize, ksize, k
            #   threshold %
            #   
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            
            dst = cv.cornerHarris(gray, blockSize = 2, ksize = 3, k = .04)
            dst = cv.dilate(dst,None)
            # Threshold for an optimal value, it may vary depending on the image.
            img[dst>0.01*dst.max()]=[255,0,0]
            fig = px.imshow(img, color_continuous_scale='gray',)
            c = dcc.Graph(figure = fig)
            return [c]
        elif cv_option == 'SIFT':
            #parameters:
            #   n/a
            gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create()
            kp = sift.detect(gray,None)
            img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            fig = px.imshow(img)
            c = dcc.Graph(figure = fig)
            return [c]
        elif cv_option == 'SURF':
            #parameters:
            #   surf threshold
            #   upright on/off
            surf = cv.xfeatures2d.SURF_create(5000)
            # set True if orientation is not a problem
            surf.setUpright(True)
            # Find keypoints and descriptors directly
            kp, des = surf.detectAndCompute(img,None)
            
            img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
            fig = px.imshow(img2)
            return [dcc.Graph(figure = fig)]
        elif cv_option == 'FAST':
            #parameters:
            #   nonmaxsuppression = T/F
            #   threshold
            # Initiate FAST object with default values
            fast = cv.FastFeatureDetector_create()
            # find and draw the keypoints
            kp = fast.detect(img,None)
            img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
            # Print all default params
            #print( "Threshold: {}".format(fast.getThreshold()) )
            #print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
            #print( "neighborhood: {}".format(fast.getType()) )
            #print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    
            # Disable nonmaxSuppression
            fast.setNonmaxSuppression(0)
            kp = fast.detect(img, None)
            #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
            img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
            fig = px.imshow(img2)
            return [dcc.Graph(figure = fig)]
        """
        elif cv_option == 'BRIEF':
            
            # Initiate FAST detector
            star = cv.xfeatures2d.StarDetector_create()
            # Initiate BRIEF extractor
            brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
            # find the keypoints with STAR
            kp = star.detect(img,None)
            # compute the descriptors with BRIEF
            kp, des = brief.compute(img, kp)
            print( brief.descriptorSize() )
            print( des.shape )
        """
        
        #return [children]
    
    return [[]]
"""
@app.callback([Output(component_id = 'altered-img-plot-holder', component_property = 'children')], 
              [Input(component_id = 'CV_Dropdown_Options', component_property = 'value')],
              [State(component_id = 'upload-image', component_property = 'contents')])

def select_mode_option(cv_option, url):
    if url is not None:
        img = cv.imread('C:\\Users\\Chris Lin\\Downloads\\Lenna.png')
        if cv_option=='HarrisCorner':
            #parameters:
            #   blocksize, ksize, k
            #   threshold %
            #   
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            
            dst = cv.cornerHarris(gray, blockSize = 2, ksize = 3, k = .04)
            dst = cv.dilate(dst,None)
            # Threshold for an optimal value, it may vary depending on the image.
            img[dst>0.01*dst.max()]=[255,0,0]
            fig = px.imshow(img, color_continuous_scale='gray',)
            c = dcc.Graph(figure = fig)
            return [c]
        elif cv_option == 'SIFT':
            #parameters:
            #   n/a
            gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create()
            kp = sift.detect(gray,None)
            img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            fig = px.imshow(img)
            c = dcc.Graph(figure = fig)
            return [c]
        elif cv_option == 'SURF':
            #parameters:
            #   surf threshold
            #   upright on/off
            surf = cv.xfeatures2d.SURF_create(5000)
            # set True if orientation is not a problem
            surf.setUpright(True)
            # Find keypoints and descriptors directly
            kp, des = surf.detectAndCompute(img,None)
            
            img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
            fig = px.imshow(img2)
            return [dcc.Graph(figure = fig)]
        elif cv_option == 'FAST':
            #parameters:
            #   nonmaxsuppression = T/F
            #   threshold
            # Initiate FAST object with default values
            fast = cv.FastFeatureDetector_create()
            # find and draw the keypoints
            kp = fast.detect(img,None)
            img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
            # Print all default params
            #print( "Threshold: {}".format(fast.getThreshold()) )
            #print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
            #print( "neighborhood: {}".format(fast.getType()) )
            #print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    
            # Disable nonmaxSuppression
            fast.setNonmaxSuppression(0)
            kp = fast.detect(img, None)
            #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
            img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
            fig = px.imshow(img2)
            return [dcc.Graph(figure = fig)]
        
        elif cv_option == 'BRIEF':
            
            # Initiate FAST detector
            star = cv.xfeatures2d.StarDetector_create()
            # Initiate BRIEF extractor
            brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
            # find the keypoints with STAR
            kp = star.detect(img,None)
            # compute the descriptors with BRIEF
            kp, des = brief.compute(img, kp)
            print( brief.descriptorSize() )
            print( des.shape )
        
        return [[]]
"""




            
#-----------------------------------------------------------------------------------
#
#function:  fills dropdown for list of available option contract dates for overall
#           fills dropdown for list of available option contract dates for greeks
#           display ticker selected; will internally update stock tracked to be the ticker 
#           fills table with company financials (off yahoo finance)
#
#initiation: press 'update ticker info' button
#
#-----------------------------------------------------------------------------------
"""
@app.callback(
    [
     Output(component_id = 'stockoverview-table', component_property = 'data'),
     ] ,
    [Input(component_id ='fill-ticker-table-info', component_property = 'n_clicks')],
    [State(component_id='ticker-input', component_property='value'),]
)
def update_output_div(n_clicks, i1, i2):
    if i1!='':
        output = i1.upper()+' with interest rate {}%'.format(i2)
    else:
        output = 'Select Ticker'
    examined_stock.update(i1, i2)
    dates = [{'label': i, 'value': i} for i in examined_stock.option_dates]
    return dates,dates,output, examined_stock.stats

"""

if __name__ == '__main__':
    app.run_server(debug=True)
    