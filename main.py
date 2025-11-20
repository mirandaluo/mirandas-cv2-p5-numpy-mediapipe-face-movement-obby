from p5 import *   #will only use for player position/velocity
import cv2               #open cv webcam and drawing
import mediapipe as mp     #face landmarks
import numpy as np      #spike polygons
import random, time, math    #utilities


#basic config

WIDTH, HEIGHT = 960, 540                 
FLOOR_Y = HEIGHT - 40   #floor Y coordinate


#face baseline state

baseline = {
    "ready": False,  #calibration finished or nah
    "frames": 0,   #frames for calibration
    "mouth": 0.0,   #mouth-open ratio should be neautral
    "yaw": 0.0    #yaw value starts as neutrak as well
}

#smoothing for numpy patterns in dataset 
smooth_mouth = 0.0                       
smooth_yaw   = 0.0                      
mouth_frames = 0     #consecutive frames where  mouth > threshold
last_jump    = 0.0    #last jump time
JUMP_COOLDOWN = 0.35 #jump cooldown

actions = {   #actions the player should take this frame should all start off as false so they dont move
    "left": False,
    "right": False,
    "jump": False
}

#MediaPipe landmarks
LEFT_EYE_OUT, RIGHT_EYE_OUT = 33, 263 #approx. outer corners of eyes
UPPER_LIP, LOWER_LIP        = 13, 14 #lips for mouth-open
NOSE_TIP                    = 1 #approx. nose tip


#helpers

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) 

def rect_overlap(a, b):
    return (
        a["x"] < b["x"] + b["w"] and
        a["x"] + a["w"] > b["x"] and
        a["y"] < b["y"] + b["h"] and
        a["y"] + a["h"] > b["y"]
    )


#player

class Player:
    def __init__(self, name):
        self.pos = Vector(80, FLOOR_Y)#p5 vector position
        self.vel = Vector(0, 0)  #p5 vector velocity
        self.on_ground = True #will be false when jumping
        self.name = name   #interactive naming

    
    # pos
    def aabb(self):
        return {"x": self.pos.x - 8,
                "y": self.pos.y - 46,
                "w": 16,
                "h": 46}

    def update(self, dt):
        speed = 220.0     #horizontal speed

        #horizontal motion from actions
        self.vel.x = 0.0 #velocity
        if actions["left"]:
            self.vel.x -= speed #movement for left
        if actions["right"]:
            self.vel.x += speed #move for right

        #gravity
        self.vel.y += 900.0 * dt 

        # jump
        if actions["jump"] and self.on_ground:
            self.vel.y = -380.0
            self.on_ground = False #not on ground anymore

        #integrate motion
        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

        #collision w floor
        if self.pos.y > FLOOR_Y:
            self.pos.y = FLOOR_Y
            self.vel.y = 0.0
            self.on_ground = True

        #screen should be horizontal
        self.pos.x = clamp(self.pos.x, 20, WIDTH - 20)

    def draw(self, frame): #stickman 
        x, y = int(self.pos.x), int(self.pos.y)
        col = (255, 255, 255)

        #body
        cv2.line(frame, (x, y-30), (x, y-6), col, 3)
        #arm
        cv2.line(frame, (x, y-24), (x-12, y-14), col, 3)
        cv2.line(frame, (x, y-24), (x+12, y-14), col, 3)
        #legs
        cv2.line(frame, (x, y-6), (x-10, y+10), col, 3)
        cv2.line(frame, (x, y-6), (x+10, y+10), col, 3)
        #head
        cv2.circle(frame, (x, y-38), 9, col, 2)
        #label for name
        cv2.putText(frame, self.name, (x-30, y-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


#obby!

class Obstacle:
    def __init__(self, x, y, w, h, kind):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.kind = kind                  #'platform', 'spikes', or 'checkpoint'

    def aabb(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def draw(self, frame):
        if self.kind == "platform":
            cv2.rectangle(frame,
                          (int(self.x), int(self.y)),
                          (int(self.x + self.w), int(self.y + self.h)),
                          (203, 192, 255), -1) #pink!
        elif self.kind == "spikes":
            step = 12
            x = self.x
            bottom = int(self.y + self.h)
            top = int(self.y)
            while x < self.x + self.w:
                pts = np.array([[x, bottom],
                                [x + step/2, top],
                                [x + step, bottom]], np.int32)
                cv2.fillConvexPoly(frame, pts, (200, 60, 60))
                x += step
        elif self.kind == "cp":
            cv2.rectangle(frame,
                          (int(self.x), int(self.y)),
                          (int(self.x + self.w), int(self.y + self.h)),
                          (80, 255, 140), -1)
            cv2.putText(frame, "NEXT",
                        (int(self.x + 10), int(self.y + self.h - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


#lvls to ts

class Level:
    def __init__(self):
        self.obs = []                     #list of obby

    def build(self):
        self.obs = []

        #ground platform
        self.obs.append(Obstacle(0, FLOOR_Y, WIDTH, 40, "platform"))

        platforms = []

        #first platform should ALWAYs be reacheble or else game is imposible
        fx = 160
        fy = FLOOR_Y - 80
        fw = random.randint(120, 150)
        self.obs.append(Obstacle(fx, fy, fw, 14, "platform"))
        platforms.append((fx, fy, fw))

        #additional platforms
        x = fx + random.randint(110, 150)
        y = fy
        for _ in range(random.randint(3, 5)):
            w = random.randint(100, 150)
            dx = random.randint(110, 170)
            dy = random.randint(-60, 60)
            x += dx
            y += dy
            y = clamp(y, 100, FLOOR_Y - 130)
            if x > WIDTH - 260:
                break
            self.obs.append(Obstacle(x, y, w, 14, "platform"))
            platforms.append((x, y, w))

        #spikes should be rando
        spike_count = random.randint(1, 2)
        candidates = []

        #ground spikes should NEVER be near first platform
        gx1 = fx + fw + 40
        gx2 = int(WIDTH * 0.75)
        candidates.append(("ground", (gx1, gx2)))

        #platform spikes only on later platforms
        if len(platforms) > 1:
            candidates.append(("platform", random.choice(platforms[1:])))
        if len(platforms) > 2:
            candidates.append(("platform", random.choice(platforms[2:])))

        for _ in range(spike_count):
            if not candidates:
                break
            kind, data = candidates.pop(random.randrange(len(candidates)))

            if kind == "ground":
                gs, ge = data
                region = ge - gs
                if region < 60:
                    continue
                maxw = min(120, region - 20)
                if maxw < 40:
                    continue
                sw = random.randint(40, maxw)
                sx = random.randint(gs, ge - sw)
                self.obs.append(Obstacle(sx, FLOOR_Y - 20, sw, 12, "spikes"))
            else:
                px, py, pw = data
                if pw < 50:
                    continue
                frac = random.uniform(0.25, 0.5)      # 25–50% of platform
                sw = int(pw * frac)
                sw = max(30, min(sw, int(pw * 0.6)))  # up to 60% of platform
                sx = px + random.uniform(0, pw - sw)
                self.obs.append(Obstacle(sx, py - 10, sw, 12, "spikes"))

        #NEXT block
        cy = FLOOR_Y - random.randint(140, 180)
        self.obs.append(Obstacle(WIDTH - 180, cy, 140, 14, "platform"))
        self.obs.append(Obstacle(WIDTH - 140, cy - 30, 70, 24, "cp"))

    def draw(self, frame):
        for o in self.obs:
            o.draw(frame)


#game class

class Game:
    def __init__(self, name):
        self.level = Level()
        self.level.build()
        self.player = Player(name)
        self.score = 0

    def reset(self):
        self.player = Player(self.player.name)

    def next_level(self):
        self.score += 1
        self.level.build()
        self.player = Player(self.player.name)

    def update(self, dt):
        self.player.update(dt)
        a = self.player.aabb()
        for o in self.level.obs:
            b = o.aabb()
            if not rect_overlap(a, b):
                continue

            if o.kind == "platform":
                prev_bottom = a["y"] + a["h"] - self.player.vel.y * dt
                if prev_bottom <= b["y"] + 2:
                    self.player.pos.y = b["y"]
                    self.player.vel.y = 0.0
                    self.player.on_ground = True

            elif o.kind == "spikes":
                self.reset()
                break

            elif o.kind == "cp":
                self.next_level()
                break

    def draw(self, frame):
        self.level.draw(frame)
        self.player.draw(frame)
        cv2.putText(frame, f"Score: {self.score}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)


#face actions are highkey broken but ppl need to face camera for ~1 sec before trynna play
def auto_calibrate_and_actions(lm):
    """
    calibration:
      -first ~45 frames: learn neutral yaw & mouth. #determine how the player stops
    control:
      - Turn head left -> move LEFT
      - Turn head RIGHT-> move RIGHT
      - Head straight-> STOP
      - Mouth open -> JUMP
    """
    global smooth_mouth, smooth_yaw, mouth_frames, last_jump 

    if lm is None:
        actions["left"] = actions["right"] = actions["jump"] = False
        return

    def P(i):
        p = lm[i]
        return (p.x * WIDTH, p.y * HEIGHT)

    pLEO = P(LEFT_EYE_OUT)
    pREO = P(RIGHT_EYE_OUT)
    pUL  = P(UPPER_LIP)
    pLL  = P(LOWER_LIP)
    pN   = P(NOSE_TIP)

    eye_span_px = dist2d(pLEO, pREO) + 1e-6

    #mouth-open normalized
    raw_mouth = abs(pUL[1] - pLL[1]) / eye_span_px

    #new yaw: nose horizontal offset relative to eye center, normalized by eye span
    center_x = 0.5 * (pLEO[0] + pREO[0])
    raw_yaw = (pN[0] - center_x) / eye_span_px  #positive: nose right of center, negative: nose left -> track the nose

    #calibration
    if not baseline["ready"]:
        if baseline["frames"] == 0:
            baseline["mouth"] = raw_mouth
            baseline["yaw"]   = raw_yaw
            smooth_mouth = raw_mouth
            smooth_yaw   = raw_yaw
        else:
            a = 0.15
            baseline["mouth"] = (1 - a) * baseline["mouth"] + a * raw_mouth
            baseline["yaw"]   = (1 - a) * baseline["yaw"]   + a * raw_yaw
            smooth_mouth = baseline["mouth"]
            smooth_yaw   = baseline["yaw"]

        baseline["frames"] += 1
        if baseline["frames"] > 45:
            baseline["ready"] = True

        actions["left"] = actions["right"] = actions["jump"] = False
        return

    #smoothing
    smooth_mouth = 0.3 * raw_mouth + 0.7 * smooth_mouth
    smooth_yaw   = 0.25 * raw_yaw   + 0.75 * smooth_yaw

    #yaw is like rotation
    yaw_delta = smooth_yaw - baseline["yaw"]
    dead = 0.12   # Slightly larger dead zone = more stable STOP

    if abs(yaw_delta) <= dead:
        actions["left"] = actions["right"] = False
    else:
        #mapping is flipped as intuitive
        #if nose is LEFT of baseline center => yaw_delta < -dead => move LEFT
        #if nose is RIGHT of baseline center=> yaw_delta >  dead => move RIGHT
        actions["left"]  = yaw_delta < -dead
        actions["right"] = yaw_delta >  dead

    #jump mouth
    now = time.time()
    thresh = max(baseline["mouth"] * 2.0, baseline["mouth"] + 0.05)

    if smooth_mouth > thresh:
        mouth_frames += 1
    else:
        mouth_frames = 0

    want = (mouth_frames >= 3)
    can  = ((now - last_jump) >= JUMP_COOLDOWN)

    if want and can:
        actions["jump"] = True
        last_jump = now
        mouth_frames = 0
    else:
        actions["jump"] = False


#main

def main():
    name = input("Enter player name: ").strip() or "Player"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found.")
        return
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    game = Game(name)
    prev = time.time()

    print("please hold your head straight ~1s for calibration.")
    print("then turn head LEFT/RIGHT to move, open mouth to jump. Q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to rgb
        results = mesh.process(rgb)
        lm = results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None

        now = time.time()
        dt = min(0.05, now - prev)
        prev = now

        auto_calibrate_and_actions(lm)
        game.update(dt)
        game.draw(frame)

        cv2.putText(frame,
                    "Turn head to move, Open mouth = jump, Q to quit",
                    (10, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1)

        cv2.imshow("project 3 - miranda obby", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
