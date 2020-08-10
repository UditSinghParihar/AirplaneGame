import pygame
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
from torch import nn
import torch
from sys import exit
import os

from pygame.locals import (
	RLEACCEL,
	K_UP,
	K_DOWN,
	K_LEFT,
	K_RIGHT,
	K_ESCAPE,
	K_q,
	KEYDOWN,
	QUIT,
)


class Player(pygame.sprite.Sprite):
	def __init__(self):
		super(Player, self).__init__()
		self.surf = pygame.image.load("asset/airplane6.png").convert()
		self.surf.set_colorkey((255, 255, 255), RLEACCEL)
		self.rect = self.surf.get_rect(
			center = (160, random.randint(0, SCREEN_HEIGHT))
		)

	def update(self, pressed_keys):
		if pressed_keys[K_UP]:
			self.rect.move_ip(0, -8)
		if pressed_keys[K_DOWN]:
			self.rect.move_ip(0, 8)
		if pressed_keys[K_LEFT]:
			self.rect.move_ip(-5, 0)
		if pressed_keys[K_RIGHT]:
			self.rect.move_ip(5, 0)

		if self.rect.left < 0:
			self.rect.left = 0
		if self.rect.right > SCREEN_WIDTH:
			self.rect.right = SCREEN_WIDTH
		if self.rect.top <= 0:
			self.rect.top = 0
		if self.rect.bottom >= SCREEN_HEIGHT:
			self.rect.bottom = SCREEN_HEIGHT


class Enemy(pygame.sprite.Sprite):
	def __init__(self):
		super(Enemy, self).__init__()
		self.surf = pygame.image.load("asset/missile4.png").convert()
		self.surf.set_colorkey((255, 255, 255), RLEACCEL)
		self.rect = self.surf.get_rect(
			center=(
				random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
				random.randint(0, SCREEN_HEIGHT),
			)
		)
		self.speed = random.randint(14, 24)
		# self.speed = random.randint(10, 15)

	def update(self):
		self.rect.move_ip(-self.speed, 0)
		if self.rect.right < 0:
			self.kill()


class Cloud(pygame.sprite.Sprite):
	def __init__(self):
		super(Cloud, self).__init__()
		self.surf = pygame.image.load("asset/cloud2.jpg").convert()
		self.surf.set_colorkey((0, 0, 0), RLEACCEL)
		self.rect = self.surf.get_rect(
			center=(
				random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
				random.randint(0, SCREEN_HEIGHT),
			)
		)

	def update(self):
		self.rect.move_ip(-5, 0)
		if self.rect.right < 0:
			self.kill()


def saveImg(screen, pressed_keys):
	global step
	global upCnt
	global downCnt
	global nothingCnt

	if(step % 4 == 0):
		if(pressed_keys[K_UP] == True):
			pygame.image.save(screen, "data3/up/frameN{:06d}.jpg".format(step))
			# print("frame{:06d}.jpg".format(step))
			upCnt = upCnt + 1

		elif(pressed_keys[K_DOWN] == True):
			pygame.image.save(screen, "data3/down/frameN{:06d}.jpg".format(step))
			# print("frame{:06d}.jpg".format(step))
			downCnt = downCnt + 1

		else:
			pygame.image.save(screen, "data3/nothing/frameN{:06d}.jpg".format(step))
			# print("frame{:06d}.jpg".format(step))
			nothingCnt = nothingCnt + 1

	step = step + 1

	if(step % 500 == 0):
		print("U: {}. D: {}. N: {}".format(upCnt, downCnt, nothingCnt))



def getModel(device, numClasses):
	model = models.resnet18(pretrained=True)
	
	model.fc = nn.Sequential(nn.Linear(512, 256), 
		nn.ReLU(),
		nn.Dropout(0.2),
		nn.Linear(256, numClasses),
		nn.LogSoftmax(dim=1))

	model.to(device)

	return model


def predictImage(img, model, device):
	testTransform = transforms.Compose([
		transforms.Resize([224, 224]), 
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	model.eval()
	with torch.no_grad():
		imgTensor = testTransform(img)
		imgTensor = imgTensor.unsqueeze_(0)
		imgTensor = imgTensor.to(device)	
		predict = model(imgTensor)
		index = predict.data.cpu().numpy().argmax()

	return index, torch.exp(predict).data.cpu().numpy().squeeze()


def takeAction(screen, pressed_keys,  model, device, classNames):
	pilStringImage = pygame.image.tostring(screen, "RGB", False)
	pilImage = Image.frombytes("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), pilStringImage)

	index, probs = predictImage(pilImage, model, device)
	action = classNames[index]

	pressed_keys_list = list(pressed_keys)		

	if(action == "down"):
		pressed_keys_list[K_DOWN] = 1
	
	elif(action == "up"):
		pressed_keys_list[K_UP] = 1

	pressed_keys = tuple(pressed_keys_list)

	return pressed_keys


if __name__ == '__main__':
	SCREEN_WIDTH = 1600
	SCREEN_HEIGHT = 1000

	os.environ['SDL_VIDEO_WINDOW_POS'] = '50,0'

	running = True
	step = 0
	upCnt = 0
	downCnt = 0
	nothingCnt = 0

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	weightName = "checkpoints/exp4/epoch9.pth"
	classNames = ['down', 'nothing', 'up']

	model = getModel(device, len(classNames))
	model.load_state_dict(torch.load(weightName))
	print("Model loaded Successfully.")

	pygame.init()

	clock = pygame.time.Clock()

	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

	ADDENEMY = pygame.USEREVENT + 1
	pygame.time.set_timer(ADDENEMY, 500)
	ADDCLOUD = pygame.USEREVENT + 2
	pygame.time.set_timer(ADDCLOUD, 1000)

	player = Player()
	enemies = pygame.sprite.Group()
	clouds = pygame.sprite.Group()
	all_sprites = pygame.sprite.Group()
	all_sprites.add(player)


	while running:
		for event in pygame.event.get():
			
			if event.type == KEYDOWN:
				if(event.key == K_ESCAPE or event.key == K_q):
					running = False		
				
			elif(event.type == ADDENEMY):
				new_enemy = Enemy()
				enemies.add(new_enemy)
				all_sprites.add(new_enemy)

			elif event.type == ADDCLOUD:
				new_cloud = Cloud()
				clouds.add(new_cloud)
				all_sprites.add(new_cloud)


		screen.fill((135, 206, 250))

		for entity in all_sprites:
			if(isinstance(entity, Cloud)):
				screen.blit(entity.surf, entity.rect)

		for entity in all_sprites:
			if(isinstance(entity, Enemy)):
				screen.blit(entity.surf, entity.rect)

		screen.blit(player.surf, player.rect)
		pressed_keys = pygame.key.get_pressed()

		# saveImg(screen, pressed_keys)
		pressed_keys = takeAction(screen, pressed_keys,  model, device, classNames)

		player.update(pressed_keys)
		enemies.update()
		clouds.update()

		if(pygame.sprite.spritecollideany(player, enemies)):
			player.kill()
			running = False

		pygame.display.flip()

		clock.tick(33)
		# print(clock.get_fps())