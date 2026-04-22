import multiprocessing as mp
import numpy as np

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    info['terminal_observation'] = ob
                    ob = env.reset(n=env.n)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset(n=data)
                remote.send(ob)
            elif cmd == 'get_legal_mask':
                mask = env.get_legal_mask()
                remote.send(mask)
            elif cmd == 'set_curriculum':
                env.set_curriculum(data)
                remote.send(True)
            elif cmd == 'get_n':
                remote.send(env.n)
            elif cmd == 'close':
                remote.close()
                break
        except EOFError:
            break
        except Exception as e:
            print(f"Worker exception: {e}")
            break

class SubprocVecEnv:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        # 使用 spawn 確保在 Windows 等平台上穩定執行
        ctx = mp.get_context('spawn')
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=worker, args=(work_remote, remote, env_fn))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()
            
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, n=None):
        for remote in self.remotes:
            remote.send(('reset', n))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)
        
    def get_legal_masks(self):
        for remote in self.remotes:
            remote.send(('get_legal_mask', None))
        masks = [remote.recv() for remote in self.remotes]
        return np.stack(masks)
        
    def set_curriculum(self, stage):
        for remote in self.remotes:
            remote.send(('set_curriculum', stage))
        for remote in self.remotes:
            remote.recv() # wait for ack
            
    def get_n(self):
        self.remotes[0].send(('get_n', None))
        return self.remotes[0].recv()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
