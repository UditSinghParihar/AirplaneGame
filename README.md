# AirplaneGame

## Learning to Fly

![Alt Text](asset/long.gif)


## Testing autopilot  
1. Clone repository:    
	1. `git clone https://github.com/UditSinghParihar/AirplaneGame.git`  
2. Install dependencies:  
	1. Install manually: `torch`, `torchvision`, `PIL`, `matplotlib`, `pygame`   
	2. Or install using `requirements.txt`: `pip install -r requirements.txt`  
3. Download weights (45MB) for google drive:  
	1. `wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YBNf3AFq4ifgx_vGlmeXRr4m9EpyZ1CL' -O best.pth`  
4. Launch game:  
	1. `python planeGame.py`  


## Training autopilot  
1. Extract images using `saveImg` function in `planeGame.py` in `data` folder:  
2. `python train.py data/ 1`  


## Todo

1. - [ ] Add bullet firing functionality to plane.  
2. - [x] Train network to perform airplane actions.  
3. - [ ] Use human pose detection to control airplane actions.  
4. - [ ] Add temporal information into the network.  