import streamlit as st
import time
import numpy as np
import tempfile
from collections import Counter
import json
import pandas as pd
from PIL import ImageColor
from src.utils.model_utils import get_system_stat
import random
from PIL import Image
from ultralytics import YOLO
import io 
import torch
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import sys
import threading
from src.config.config import load_config
import asyncio

# Import cv2 inside a try-except block to handle potential import issues
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    st.error("OpenCV could not be imported. Some features may not be available.")
    OPENCV_AVAILABLE = False

    
def get_yolo(img, model, confidence, color_pick_list, class_labels, draw_thick):
    """Perform object detection, draw bounding boxes, and black out glasses."""
    current_no_class = []
    results = model.predict(img)

    for result in results:
        bboxs = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])

            # Check if confidence is above the threshold
            if cnf > confidence:
                # Black out the glasses region
                img[ymin:ymax, xmin:xmax] = 0  # Black out the region

                # Draw bounding box and label
                plot_one_box([xmin, ymin, xmax, ymax], img, label=class_labels[int(cs)],
                             color=color_pick_list[int(cs)], line_thickness=draw_thick)

                # Append detected class
                current_no_class.append([class_labels[int(cs)]])
    global last_detections  # For stats display
    last_detections = dict(Counter(i for sub in current_no_class for i in set(sub)))
    
    return img, current_no_class

def color_picker_fn(classname, key):
    color_picke = st.sidebar.color_picker(f'{classname}:', '#ff0003', key=key)
    color_rgb_list = list(ImageColor.getcolor(str(color_picke), "RGB"))
    color = [color_rgb_list[0], color_rgb_list[1],color_rgb_list[2] ]
    return color

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """Draw a single bounding box on the image."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def detect_objects(image, model, confidence, color_pick_list, class_labels, draw_thick):
    """Perform object detection and draw bounding boxes."""
    results = model.predict(image)
    current_no_class = []

    for result in results:
        bboxs = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        for bbox, cnf, cs in zip(bboxs, conf, cls):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            if cnf > confidence:
                plot_one_box([xmin, ymin, xmax, ymax], image, label=class_labels[int(cs)],
                             color=color_pick_list[int(cs)], line_thickness=draw_thick)
                current_no_class.append([class_labels[int(cs)]])
    return image, current_no_class

def run_app(): 
   
    lock = threading.Lock()
    img_container = {"img": None}

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        with lock:
            img_container["img"] = img

        return frame
   
    p_time = 0
    
    # Set page configuration with a professional look
    st.set_page_config(
        page_title="Mobile Tech Industries - CV Detection", 
        layout="wide", 
        initial_sidebar_state="expanded",
        page_icon="📱"
    )

    # Custom CSS for a more professional look
       # Custom CSS for a blue and white theme
    st.markdown("""
    <style>
        /* Main content styling */
        .main {
            background-color: #f0f8ff;  /* Light blue background */
            padding: 20px;
        }
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #1a4b8c;  /* Darker blue for sidebar */
            color: white !important;
        }
        
        /* Make ALL text in sidebar white */
        [data-testid="stSidebar"] .st-emotion-cache-183lzff {
            color: white !important;
        }
        
        /* Sidebar text elements */
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stRadio label {
            color: white !important;
        }
        
        /* Selectbox labels specifically */
        [data-testid="stSidebar"] .stSelectbox > div > label {
            color: white !important;
        }
        
        /* Dropdown options */
        div[data-baseweb="select"] ul li {
            color: #1a4b8c !important;  /* Blue text for dropdown options */
        }
        
        /* Selected option in the selectbox - make it blue */
        div[data-baseweb="select"] span {
            color: #1a4b8c !important;  /* Blue text for the selected value */
        }
        
        /* Make dropdown text blue when dropdown is open */
        div[role="listbox"] span {
            color: #1a4b8c !important;
        }
        
        /* File uploader label */
        [data-testid="stSidebar"] .stFileUploader label {
            color: white !important;
        }
        
        /* Slider labels and values */
        [data-testid="stSidebar"] .stSlider div {
            color: white !important;
        }
        
        /* Header styling */
        h1, h2, h3, h4 {
            font-family: 'Arial', sans-serif;
            color: #0046ad;  /* Blue headers */
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #0062cc;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #004a99;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        
        /* Card-like containers */
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(25, 75, 151, 0.15);
            border-left: 4px solid #0062cc;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Status indicators */
        .status-active {
            color: #0073e6;
            font-weight: bold;
        }
        
        .status-inactive {
            color: #dc3545;
            font-weight: bold;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #b3d1ff;
            font-size: 0.8em;
            color: #004a99;
        }
        
        /* Improved metric displays */
        .metric-container {
            background-color: #e6f0ff;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 10px;
            border: 1px solid #b3d1ff;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #0062cc;
        }
        
        .metric-label {
            color: #004a99;
            font-size: 0.9em;
        }
        
        /* Hide MainMenu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom header styling */
        .company-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #b3d1ff;
        }
        
        .app-header {
            font-size: 2.2em;
            font-weight: bold;
            background: linear-gradient(90deg, #0046ad, #0073e6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .app-subheader {
            color: #005cbf;
            margin-top: -5px;
        }
        
        /* Table styling */
        .dataframe {
            border: 1px solid #b3d1ff;
        }
        
        .dataframe th {
            background-color: #e6f0ff;
            color: #004a99;
        }
        
        /* Slider accent colors */
        .stSlider > div > div > div > div {
            background-color: #0062cc !important;
        }
        
        /* Radio buttons */
        .stRadio > div:first-of-type > div[role="radiogroup"] > label > div:first-of-type {
            background-color: #0062cc !important;
        }
        
        /* Checkbox */
        .stCheckbox > div > label > div:first-of-type {
            background-color: #0062cc !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Add company logo to sidebar
    with st.sidebar:
        try:
            logo = Image.open("src/mt.png")
            st.image(logo, width=180)
        except FileNotFoundError:
            st.sidebar.error("Logo file (mt.png) not found")
    
    # Header with company branding
    st.markdown("""
    <div class="company-header">
        <div>
            <div class="app-header">Glasses Detection App</div>
            <div class="app-subheader">Advanced Computer Vision Detection Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <p>Experience glasses detection powered by state-of-the-art AI models. 
        Our solution helps protect user privacy by automatically identifying and obscuring glasses in images and video streams.</p>
    </div>
    """, unsafe_allow_html=True)
            
    # Sidebar configuration
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #495057;">
        <h3 style="color: #f8f9fa;">Detection Settings</h3>
    </div>
    """, unsafe_allow_html=True)

    # Choose the model
    model_type = st.sidebar.selectbox(
        'Choose Detection Model', ('Select Model','Glasses Detection', 'Upload Model')
    )

    cap = None

    if model_type == 'Upload Model':
        path_model_file = st.sidebar.text_input(
            f'Path to {model_type}:',
            f'eg: models/best.pt'
        )

    # YOLOv11 Model
    elif model_type == 'Glasses Detection':
        # Load configuration
        config_path = 'configs/app_config.yaml'
        config = load_config(config_path)

        # Use the model path from the configuration file
        path_model_file = config['inference']['path']

    elif model_type == 'Select Model':
        pass
    else:
        st.error('Model not found!')

    # Main content area with cards
    st.markdown("""
    <div class="card">
        <h2>Detection Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(f'{model_type} Predictions')
    col1, col2 = st.columns([2,2])
    org_frame = col1.empty()
    ann_frame = col2.empty()

    pred = False
    pred1 = False
    pred2 = False
    if model_type!= 'Select Model':
        
        load = st.sidebar.checkbox("Load Model", key='Load Model')

        if load:
            with st.spinner("Loading model..."):
                try:
                    model = YOLO(path_model_file)
                    print("Model loaded successfully!")
                    st.sidebar.markdown("""
                    <div class="status-active">✓ Model loaded successfully!</div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    print(f"Failed to load model: {e}")
                    st.sidebar.markdown("""
                    <div class="status-inactive">✗ Failed to load model!</div>
                    """, unsafe_allow_html=True)
            
            # Load Class names
            class_labels = model.names
            
            # Inference Mode
            st.sidebar.markdown("""
            <div style="margin-top: 20px; margin-bottom: 10px;">
                <h4 style="color: #f8f9fa;">Input Source</h4>
            </div>
            """, unsafe_allow_html=True)
            
            options = st.sidebar.radio('Select input source:', ('Image', 'Video', 'Webcam'), key='options')

            # Confidence
            st.sidebar.markdown("""
            <div style="margin-top: 20px; margin-bottom: 10px;">
                <h4 style="color: #f8f9fa;">Detection Parameters</h4>
            </div>
            """, unsafe_allow_html=True)
            
            confidence = st.sidebar.slider( 
                'Detection Confidence', 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6,
                key='confidence'
            )

            # Draw thickness
            draw_thick = st.sidebar.slider(
                'Box Thickness:', 
                min_value=1,
                max_value=20, 
                value=2, 
                key='draw_thick'
            )
            
            # Color picker section
            st.sidebar.markdown("""
            <div style="margin-top: 20px; margin-bottom: 10px;">
                <h4 style="color: #f8f9fa;">Visualization Colors</h4>
            </div>
            """, unsafe_allow_html=True)
            
            color_pick_list = []
            color_rev_list = []
            for i in range(len(class_labels)):
                classname = class_labels[i]
                color = color_picker_fn(classname, i)
                color_rev = color[::-1]
                color_pick_list.append(color)
                color_rev_list.append(color_rev)
            
            # Image
            if options == 'Image':
                st.sidebar.markdown("""
                <div style="margin-top: 20px; margin-bottom: 10px;">
                    <h4 style="color: #f8f9fa;">Upload Image</h4>
                </div>
                """, unsafe_allow_html=True)
                             
                upload_img_file = st.sidebar.file_uploader('Select image file', type=['jpg', 'jpeg', 'png'], key='image_uploader')
                
                if upload_img_file is not None:
                    pred = st.button("Process Image")
                    byte_data = upload_img_file.read()
                    image = Image.open(upload_img_file)
                    img = np.array(image)
                    org_frame.image(upload_img_file, caption='Original Image', channels="BGR", use_container_width=True)

                    if pred:
                        with st.spinner("Processing image..."):
                            img, current_no_class = detect_objects(img, model, confidence, color_pick_list, class_labels, draw_thick)    
                            ann_frame.image(img, caption='Processed Image (Glasses Obscured)', channels="RGB", use_container_width=True)
                            
                            # Current number of classes
                            class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                            class_fq = json.dumps(class_fq, indent = 4)
                            class_fq = json.loads(class_fq)
                            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Quantity'])
                            
                            # Updating Inference results
                            with st.container():
                                st.markdown("""
                                <div class="card">
                                    <h3>Detection Statistics</h3>
                                    <p>Objects identified in the current image:</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.dataframe(df_fq, use_container_width=True)

            # Video
            elif options == 'Video':
                st.sidebar.markdown("""
                <div style="margin-top: 20px; margin-bottom: 10px;">
                    <h4 style="color: #f8f9fa;">Upload Video</h4>
                </div>
                """, unsafe_allow_html=True)
                
                upload_video_file = st.sidebar.file_uploader('Select video file', type=['mp4', 'avi', 'mkv'], key='vid_uploader')
                
                if upload_video_file is not None:
                    g = io.BytesIO(upload_video_file.read()) # BytesIO Object
                    vid_location = "ultralytics.mp4"
                    with open(vid_location, "wb") as out:  # Open temporary file as bytes
                        out.write(g.read())  # Read bytes into file
                    vid_file_name = "ultralytics.mp4"
                    pred1 = st.sidebar.button("Process Video")
                    cap = cv2.VideoCapture(vid_file_name)
                   
            # Web-cam
            elif options == 'Webcam':
                st.sidebar.markdown("""
                <div style="margin-top: 20px; margin-bottom: 10px;">
                    <h4 style="color: #f8f9fa;">Webcam Settings</h4>
                </div>
                """, unsafe_allow_html=True)
                
                class VideoProcessor(VideoTransformerBase):
                    def __init__(self):
                        self.model = model
                        self.confidence = confidence
                        self.color_rev_list = color_rev_list
                        self.class_labels = class_labels
                        self.draw_thick = draw_thick
                        
                    def transform(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Process frame with YOLO
                        processed_img, _ = get_yolo(
                            img.copy(), 
                            self.model, 
                            self.confidence, 
                            self.color_rev_list, 
                            self.class_labels, 
                            self.draw_thick
                        )
                        
                        return processed_img
                
                webrtc_ctx = webrtc_streamer(
                    key="glasses-detection",
                    video_processor_factory=VideoProcessor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                
                if webrtc_ctx.video_processor:
                    # Display detection stats below the stream
                    with st.expander("Detection Statistics"):
                        if hasattr(webrtc_ctx.video_processor, "last_detections"):
                            df_fq = pd.DataFrame(
                                webrtc_ctx.video_processor.last_detections.items(),
                                columns=['Class', 'Quantity']
                            )
                            st.dataframe(df_fq)
            if pred1 and (cap is not None):
                class_names = list(model.names.values())
                selected_classes = st.sidebar.multiselect("Classes to detect", class_names, default=class_names[:3], key='select_class')
                        
                with st.spinner("Processing video..."):
                    fps_display = st.sidebar.empty()
                    
                    if not cap.isOpened():
                        st.error("Could not open video file.")
                    
                    stop_button = st.button("Stop Processing")

                    while True: 
                        success, frame = cap.read()
                        if not success:
                            st.success("Video processing completed.")
                            break
                        
                        # Display original frame
                        org_frame.image(frame, caption="Original Video", channels="BGR", use_container_width=True)
                        
                        # Process frame
                        img, current_no_class = get_yolo(frame.copy(), model, confidence, color_rev_list, class_labels, draw_thick)
                        
                        # Display processed frame
                        ann_frame.image(img, caption="Processed Video (Glasses Obscured)", channels="BGR", use_container_width=True)

                        # FPS calculation
                        c_time = time.time()
                        fps = 1 / (c_time - p_time) if p_time > 0 else 30.0
                        p_time = c_time
                    
                        # Detection statistics
                        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                        class_fq = json.dumps(class_fq, indent = 1)
                        class_fq = json.loads(class_fq)
                        df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Quantity'])
                        
                        # Display FPS
                        fps_display.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{fps:.1f}</div>
                            <div class="metric-label">FPS</div>
                        </div>
                        """, unsafe_allow_html=True)
                            
                        if stop_button:
                            cap.release()
                            torch.cuda.empty_cache()
                            st.stop()

                    cap.release()
                    torch.cuda.empty_cache()
                    cv2.destroyAllWindows()

    # Add footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Mobile Technology Industry. All rights reserved.</p>
        <p>Privacy Detection Platform v1.0 | Enterprise Edition</p>
    </div>
    """, unsafe_allow_html=True)

# Main function call
if __name__ == "__main__":
    run_app()