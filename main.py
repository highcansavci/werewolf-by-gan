import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset.colorization_dataset import make_dataloader
from model.main_model import MainModel

if __name__ == "__main__":
    cap = cv2.VideoCapture('the_lighthouse.mp4')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_model = MainModel()
    main_model.load_state_dict(torch.load("gan_colorization.pth", map_location=device))
    if not cap.isOpened():
        print("Error opening video stream or file")

    cap.set(3, 640)
    cap.set(4, 640)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('the_lighthouse_colorized.mp4', fourcc, 30, (640, 640))

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imwrite(f"valid_path/frame.jpg", frame)
        if ret:
            dataloader = make_dataloader()
            for data in tqdm(dataloader):
                colorized_frame = main_model.visualize(data)
                colorized_frame = (cv2.resize(colorized_frame, (640, 640)) * 255).astype(np.uint8)
                colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_RGB2BGR)
                out.write(colorized_frame)

    # When everything done, release the video capture object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
