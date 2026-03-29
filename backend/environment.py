"""Blob Arena Game Environment"""
import numpy as np
import math

ARENA_W = 800        
ARENA_H = 600
AGENT_SPEED_SLOW = 3.0
AGENT_SPEED_FAST = 6.0
AGENT_RADIUS     = 16
PREY_SPEED       = 2.0
PREY_RADIUS      = 10
PREY_COUNT       = 50
CATCH_DIST = AGENT_RADIUS + PREY_RADIUS + 6
CORNER_DIST      = 80                            
WALL_PUNISH_DIST = 20                           

K_NEAREST        = 5   

# 8 dirs × 2 speeds + idle = 17 actions
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

def action_to_velocity(action: int):
    """Returns (vx, vy) for a given action index."""
    speed = AGENT_SPEED_SLOW if action < 8 else AGENT_SPEED_FAST
    angle_deg = ANGLES[action % 8]
    angle_rad = math.radians(angle_deg)
    vx = speed * math.cos(angle_rad)
    vy = speed * math.sin(angle_rad)
    return (vx, vy)


class Prey:
    def __init__(self, x, y):
        self.x  = float(x)
        self.y  = float(y)
        self.vx = 0.0
        self.vy = 0.0

    def update(self, agent_x, agent_y, all_prey):
        # flee agent, separate from neighbours, random drift
        DETECTION_R  = 180
        SEPARATION_R = 40
        FLEE_W       = 2.0
        SEP_W        = 1.0
        RAND_W       = 0.3

        fx, fy = 0.0, 0.0

        # 1. Flee agent
        dx = self.x - agent_x
        dy = self.y - agent_y
        dist = math.hypot(dx, dy)
        if dist < DETECTION_R and dist > 0:
            fx += FLEE_W * dx / dist
            fy += FLEE_W * dy / dist

        # 2. Separate from neighbours
        for other in all_prey:
            if other is self:
                continue
            dx2 = self.x - other.x
            dy2 = self.y - other.y
            d2 = math.hypot(dx2, dy2)
            if 0 < d2 < SEPARATION_R:
                fx += SEP_W * dx2 / d2
                fy += SEP_W * dy2 / d2

        # 3. Random drift
        fx += RAND_W * (np.random.rand() * 2 - 1)
        fy += RAND_W * (np.random.rand() * 2 - 1)

        # Blend into velocity (inertia)
        self.vx = 0.7 * self.vx + 0.3 * fx
        self.vy = 0.7 * self.vy + 0.3 * fy

        # Clamp speed
        spd = math.hypot(self.vx, self.vy)
        if spd > PREY_SPEED:
            self.vx = self.vx / spd * PREY_SPEED
            self.vy = self.vy / spd * PREY_SPEED

        # Move
        self.x += self.vx
        self.y += self.vy

        # Bounce off walls
        if self.x < PREY_RADIUS * 2:
            self.x = PREY_RADIUS * 2
            self.vx *= -1
        if self.x > ARENA_W - PREY_RADIUS * 2:
            self.x = ARENA_W - PREY_RADIUS * 2
            self.vx *= -1
        if self.y < PREY_RADIUS * 2:
            self.y = PREY_RADIUS * 2
            self.vy *= -1
        if self.y > ARENA_H - PREY_RADIUS * 2:
            self.y = ARENA_H - PREY_RADIUS * 2
            self.vy *= -1

class BlobArena:

    def __init__(self, seed=None):
        self.rng   = np.random.default_rng(seed)
        self.steps = 0
        self.MAX_STEPS = 2000    
        self.reset()

    def reset(self):
        """Randomise positions, return initial state vector."""
        self.steps = 0

        # Agent starts near centre with small random offset
        self.agent_x  = ARENA_W / 2 + self.rng.uniform(-50, 50)
        self.agent_y  = ARENA_H / 2 + self.rng.uniform(-50, 50)
        self.agent_vx = 0.0
        self.agent_vy = 0.0

        # Prey scattered randomly but not too close to agent at start
        self.prey = []
        attempts  = 0
        while len(self.prey) < PREY_COUNT and attempts < 1000:
            px = self.rng.uniform(PREY_RADIUS, ARENA_W - PREY_RADIUS)
            py = self.rng.uniform(PREY_RADIUS, ARENA_H - PREY_RADIUS)
            if math.hypot(px - self.agent_x, py - self.agent_y) > 120:
                self.prey.append(Prey(px, py))
            attempts += 1

        return self._get_state()

    def step(self, action: int):
        assert 0 <= action <= 16, f"Invalid action {action}"
        self.steps += 1

        # Move agent
        vx, vy = action_to_velocity(action)
        self.agent_vx = vx
        self.agent_vy = vy
        self.agent_x  = np.clip(self.agent_x + vx, AGENT_RADIUS, ARENA_W - AGENT_RADIUS)
        self.agent_y  = np.clip(self.agent_y + vy, AGENT_RADIUS, ARENA_H - AGENT_RADIUS)

        # Move prey (they react to agent position BEFORE catch check)
        for p in self.prey:
            p.update(self.agent_x, self.agent_y, self.prey)

        # Catch check
        caught = []
        for p in self.prey:
            if math.hypot(p.x - self.agent_x, p.y - self.agent_y) < CATCH_DIST:
                caught.append(p)
        for p in caught:
            self.prey.remove(p)

        # Reward
        reward, reward_info = self._compute_reward(
            caught_count=len(caught),
            action=action
        )

        # Terminal conditions
        done = (len(self.prey) == 0) or (self.steps >= self.MAX_STEPS)

        next_state = self._get_state()
        return next_state, reward, done, reward_info

    def _compute_reward(self, caught_count: int, action: int):
        r = {}

        r['catch'] = 10.0 * caught_count
        r['time']  = -0.01

        # dense signal before first catch
        min_dist = self._min_prey_distance()
        if min_dist is not None and min_dist > 0:
            r['proximity'] = min(0.05 / min_dist * (ARENA_W / 2), 1.0)    # normalised
        else:
            r['proximity'] = 0.0

        hit_wall = (
            self.agent_x <= AGENT_RADIUS + WALL_PUNISH_DIST or
            self.agent_x >= ARENA_W - AGENT_RADIUS - WALL_PUNISH_DIST or
            self.agent_y <= AGENT_RADIUS + WALL_PUNISH_DIST or
            self.agent_y >= ARENA_H - AGENT_RADIUS - WALL_PUNISH_DIST
        )
        r['wall']   = -0.5 if hit_wall else 0.0
        r['corner'] = self._corner_bonus()

        total = sum(r.values())
        r['total'] = total
        return total, r

    def _min_prey_distance(self):
        if not self.prey:
            return None
        return min(math.hypot(p.x - self.agent_x, p.y - self.agent_y) for p in self.prey)

    def _corner_bonus(self):
        # reward cornering prey near walls
        bonus = 0.0
        for p in self.prey:
            near_wall = (
                p.x < CORNER_DIST or p.x > ARENA_W - CORNER_DIST or
                p.y < CORNER_DIST or p.y > ARENA_H - CORNER_DIST
            )
            close_to_agent = math.hypot(p.x - self.agent_x, p.y - self.agent_y) < CATCH_DIST * 2
            if near_wall and close_to_agent:
                bonus += 0.3
        return bonus

    def _get_state(self):
        # flat float32 array, length 4 + 3*K_NEAREST + 4 = 19, all normalised to [0,1]
        state = np.zeros(4 + 3 * K_NEAREST + 4, dtype=np.float32)

        # Agent position & velocity
        state[0] = self.agent_x / ARENA_W
        state[1] = self.agent_y / ARENA_H
        state[2] = self.agent_vx / AGENT_SPEED_FAST
        state[3] = self.agent_vy / AGENT_SPEED_FAST

        # K-nearest prey, pad with far-away defaults if fewer remain
        sorted_prey = sorted(
            self.prey,
            key=lambda p: math.hypot(p.x - self.agent_x, p.y - self.agent_y)
        )
        for i in range(K_NEAREST):
            offset = 4 + i * 3
            if i < len(sorted_prey):
                p = sorted_prey[i]
                rel_x = (p.x - self.agent_x) / ARENA_W + 0.5
                rel_y = (p.y - self.agent_y) / ARENA_H + 0.5
                dist  = math.hypot(p.x - self.agent_x, p.y - self.agent_y) / math.hypot(ARENA_W, ARENA_H)
                state[offset]     = np.clip(rel_x, 0, 1)
                state[offset + 1] = np.clip(rel_y, 0, 1)
                state[offset + 2] = dist
            else:
                state[offset]     = 0.5
                state[offset + 1] = 0.5
                state[offset + 2] = 1.0

        # Wall distances
        state[13] = self.agent_y / ARENA_H                              # N
        state[14] = (ARENA_H - self.agent_y) / ARENA_H                  # S
        state[15] = (ARENA_W - self.agent_x) / ARENA_W                  # E
        state[16] = self.agent_x / ARENA_W                              # W

        return state

    def to_json(self):
        return {
            "agent": {"x": round(self.agent_x, 1), "y": round(self.agent_y, 1)},
            "prey":  [{"x": round(p.x, 1), "y": round(p.y, 1)} for p in self.prey],
            "steps": self.steps,
            "prey_remaining": len(self.prey),
        }


if __name__ == "__main__":

    env   = BlobArena(seed=42)
    state = env.reset()

    print(f"\nState vector length : {len(state)}  (expected 17)")
    print(f"State sample        : {state.round(3)}")
    print(f"Prey count          : {len(env.prey)}")
    print()

    total_reward  = 0
    total_catches = 0
    reward_log    = {"catch": 0, "time": 0, "proximity": 0, "wall": 0, "corner": 0}

    for step in range(500):
        action = np.random.randint(0, 17)    # random policy
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        for k in reward_log:
            reward_log[k] += info.get(k, 0)

        if info["catch"] > 0:
            total_catches += int(info["catch"] / 5.0)
            print(f"  Step {step:4d} | CATCH! prey remaining: {len(env.prey)}")

        if done:
            print(f"\n  Episode ended at step {step} (prey left: {len(env.prey)})")
            state = env.reset()

    print("\n── Reward breakdown over 500 steps ──")
    for k, v in reward_log.items():
        print(f"  {k:12s}: {v:+.2f}")
    print(f"  {'TOTAL':12s}: {total_reward:+.2f}")
    print(f"\nTotal catches : {total_catches}")