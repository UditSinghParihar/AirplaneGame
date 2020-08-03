import pygame
import random


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
		# self.surf = pygame.Surface((75, 25))
		# self.surf.fill((255, 255, 255))
		self.surf = pygame.image.load("asset/airplane6.png").convert()
		self.surf.set_colorkey((255, 255, 255), RLEACCEL)
		self.rect = self.surf.get_rect()

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
		# self.surf = pygame.Surface((20, 10))
		# self.surf.fill((255, 255, 255))
		self.surf = pygame.image.load("asset/missile4.png").convert()
		self.surf.set_colorkey((255, 255, 255), RLEACCEL)
		self.rect = self.surf.get_rect(
			center=(
				random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
				random.randint(0, SCREEN_HEIGHT),
			)
		)
		self.speed = random.randint(14, 24)

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

	if(pressed_keys[K_UP] == True):
		pygame.image.save(screen, "data/up/frameC{:06d}.jpg".format(step))
		# print("frame{:06d}.jpg".format(step))
		upCnt = upCnt + 1

	elif(pressed_keys[K_DOWN] == True):
		pygame.image.save(screen, "data/down/frameC{:06d}.jpg".format(step))
		# print("frame{:06d}.jpg".format(step))
		downCnt = downCnt + 1

	else:
		pygame.image.save(screen, "data/nothing/frameC{:06d}.jpg".format(step))
		# print("frame{:06d}.jpg".format(step))
		nothingCnt = nothingCnt + 1

	step = step + 1

	if(step % 500 == 0):
		print("Up count: {}. Down count: {}. Nothing count: {}".format(upCnt, downCnt, nothingCnt))



if __name__ == '__main__':
	SCREEN_WIDTH = 1600
	SCREEN_HEIGHT = 1000

	running = True
	step = 0
	upCnt = 0
	downCnt = 0
	nothingCnt = 0

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

		pressed_keys = pygame.key.get_pressed()
		player.update(pressed_keys)
		enemies.update()
		clouds.update()

		screen.fill((135, 206, 250))

		for entity in all_sprites:
			if(isinstance(entity, Cloud)):
				screen.blit(entity.surf, entity.rect)

		for entity in all_sprites:
			if(isinstance(entity, Enemy)):
				screen.blit(entity.surf, entity.rect)

		screen.blit(player.surf, player.rect)

		# saveImg(screen, pressed_keys)


		if(pygame.sprite.spritecollideany(player, enemies)):
			player.kill()
			running = False

		pygame.display.flip()

		clock.tick(30)