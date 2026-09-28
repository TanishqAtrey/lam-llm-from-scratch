import json
import random
import os
import tempfile
from .templates import TEMPLATES, sample_goal, TITLE_POOL, TAG_POOL, DUE_POOL, PRIO_POOL
from .oracle import Oracle
from kanha.lam.environment import TaskEnvironment
from kanha.lam.task_store import TaskStore

class TrajectoryGenerator:
    def __init__(self, num_trajectories=100_000, tier_weights=None, seed=42, reference_date=None):
        self.num_trajectories = num_trajectories
        self.rng = random.Random(seed)
        self.reference_date = reference_date
        
        if tier_weights is None:
            self.tier_weights = {
                'tier1': 0.20,
                'tier2': 0.15,
                'tier3': 0.20,
                'tier4': 0.10,
                'tier5': 0.10,
                'tier6': 0.10,
                'tier7': 0.10,
                'tier8': 0.05
            }
        else:
            self.tier_weights = tier_weights
            
    def _sample_world(self, rng, min_size=0, max_size=12, require_titles=None) -> list[dict]:
        size = rng.randint(min_size, max_size)
        tasks = []
        
        if require_titles:
            for title in require_titles:
                tasks.append({
                    'title': title,
                    'due': rng.choice(DUE_POOL),
                    'prio': rng.choice(PRIO_POOL),
                    'tag': rng.choice(TAG_POOL)
                })
        
        while len(tasks) < size:
            tasks.append({
                'title': rng.choice(TITLE_POOL),
                'due': rng.choice(DUE_POOL),
                'prio': rng.choice(PRIO_POOL),
                'tag': rng.choice(TAG_POOL)
            })
            
        return tasks
        
    def generate(self, output_path: str) -> dict:
        oracle = Oracle()
        tiers = list(self.tier_weights.keys())
        weights = list(self.tier_weights.values())
        
        per_tier = {t: 0 for t in tiers}
        total = 0
        skipped = 0
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Use a temp directory for ephemeral task stores during generation
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i in range(self.num_trajectories):
                    tier = self.rng.choices(tiers, weights=weights)[0]
                    
                    # Create an ephemeral store for sampling the goal
                    tmp_store_path = os.path.join(tmpdir, f"store_{i}.json")
                    store = TaskStore(path=tmp_store_path, reference_date=self.reference_date)
                    
                    # For tier 7, we need duplicate/ambiguous titles
                    require_titles = None
                    if tier == 'tier7':
                        title = self.rng.choice(TITLE_POOL)
                        require_titles = [title, title]
                        
                    world_tasks = self._sample_world(self.rng, require_titles=require_titles)
                    for t in world_tasks:
                        store.add(title=t['title'], due=t['due'], prio=t['prio'], tag=t['tag'])
                        
                    goal, info = sample_goal(tier, store, self.rng)
                    
                    # Create a fresh environment for the oracle to solve
                    tmp_env_path = os.path.join(tmpdir, f"env_{i}.json")
                    env = TaskEnvironment(task_store_path=tmp_env_path, reference_date=self.reference_date)
                    env.reset(initial_tasks=world_tasks)
                    
                    try:
                        steps_tuples = oracle.solve(info, env)
                    except Exception:
                        skipped += 1
                        continue
                    
                    if not steps_tuples:
                        skipped += 1
                        continue
                    
                    steps = [{'action': act, 'observation': obs} for act, obs in steps_tuples]
                    
                    traj = {
                        'goal': goal,
                        'world': world_tasks,
                        'steps': steps,
                        'tier': info['tier'],
                        'template_id': f"{tier}_{info.get('solver', 'unknown')}_{i}"
                    }
                    
                    f.write(json.dumps(traj) + '\n')
                    per_tier[tier] += 1
                    total += 1
                    
        return {'total': total, 'skipped': skipped, 'per_tier': per_tier}

        
def split_data(input_path, train_path, val_path, test_path, val_ratio=0.1, test_ratio=0.1, holdout_templates=0.2, seed=42):
    rng = random.Random(seed)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    trajectories = [json.loads(line) for line in lines]
    
    templates = list(set(t['template_id'] for t in trajectories))
    rng.shuffle(templates)
    
    holdout_idx = int(len(templates) * (1.0 - holdout_templates))
    seen_templates = set(templates[:holdout_idx])
    unseen_templates = set(templates[holdout_idx:])
    
    train, val, test = [], [], []
    
    for t in trajectories:
        if t['template_id'] in unseen_templates:
            if rng.random() < 0.5:
                val.append(t)
            else:
                test.append(t)
        else:
            r = rng.random()
            if r < val_ratio:
                val.append(t)
            elif r < val_ratio + test_ratio:
                test.append(t)
            else:
                train.append(t)
                
    for path, data in [(train_path, train), (val_path, val), (test_path, test)]:
        with open(path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
                
    return {
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test)
    }
