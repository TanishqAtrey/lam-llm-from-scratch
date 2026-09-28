import random
import re

TITLE_POOL = [
    'buy milk', 'call dentist', 'submit report', 'pick up kids', 'pay electricity bill',
    'schedule meeting with Alex', 'book flight tickets', 'renew passport', 'clean garage',
    'water the plants', 'buy groceries', 'call mom', 'feed the dog', 'take out trash',
    'pay rent', 'email boss', 'update resume', 'do laundry', 'wash car', 'mow lawn',
    'buy birthday gift', 'cancel subscription', 'call plumber', 'schedule haircut',
    'go to gym', 'read a book', 'meditate', 'buy coffee beans', 'change oil',
    'book hotel', 'file taxes', 'review pr', 'write tests', 'fix bug', 'deploy app',
    'call internet provider', 'buy new phone', 'sell old laptop', 'clean windows',
    'organize desk', 'buy stamps', 'mail package', 'sign contract', 'renew insurance',
    'buy concert tickets', 'cancel meeting', 'reschedule appointment', 'call doctor',
    'buy medicine', 'pick up prescription', 'go for a run', 'do yoga', 'buy shoes',
    'donate clothes', 'clean fridge', 'defrost freezer', 'buy batteries', 'change lightbulb',
    'fix leaky faucet', 'call electrician', 'paint bedroom', 'buy furniture', 'assemble desk',
    'buy plant', 'fertilize lawn', 'trim hedges', 'clean gutters', 'sweep driveway',
    'buy vacuum', 'empty dishwasher', 'load dishwasher', 'cook dinner', 'meal prep',
    'buy spices', 'bake cake', 'buy wine', 'host dinner party', 'send invitations',
    'buy paper towels', 'buy toilet paper', 'clean bathroom', 'scrub tub', 'buy soap',
    'restock pantry', 'buy dog food', 'walk the dog', 'take cat to vet', 'buy cat litter',
    'clean litter box', 'buy fish food', 'clean fish tank', 'buy bird seed', 'feed birds',
    'buy hamster food', 'clean cage', 'buy guinea pig food', 'buy hay', 'buy treats',
    'schedule vet appointment', 'buy flea medicine', 'give dog a bath', 'brush dog',
    'trim dog nails', 'buy dog toys', 'replace dog bed', 'buy leash', 'buy collar',
    'register dog', 'get dog license', 'buy pet insurance', 'cancel pet insurance',
    'buy cat toys', 'replace scratching post'
]

TAG_POOL = ['home', 'work', 'health', 'errands', 'finance', 'personal', 'school', 'shopping']

DUE_POOL = ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'next week']

PRIO_POOL = ['low', 'med', 'high']

DUE_PHRASES = {
    'today': ['today', 'by today', 'for today', 'due today'],
    'tomorrow': ['tomorrow', 'by tomorrow', 'for tomorrow', 'due tomorrow'],
    'monday': ['monday', 'on monday', 'by monday', 'this monday', 'for monday'],
    'tuesday': ['tuesday', 'on tuesday', 'by tuesday', 'this tuesday', 'for tuesday'],
    'wednesday': ['wednesday', 'on wednesday', 'by wednesday', 'this wednesday', 'for wednesday'],
    'thursday': ['thursday', 'on thursday', 'by thursday', 'this thursday', 'for thursday'],
    'friday': ['friday', 'on friday', 'by friday', 'this friday', 'for friday'],
    'saturday': ['saturday', 'on saturday', 'by saturday', 'this saturday', 'for saturday'],
    'sunday': ['sunday', 'on sunday', 'by sunday', 'this sunday', 'for sunday'],
    'next week': ['next week', 'by next week', 'for next week', 'due next week']
}

PRIO_PHRASES = {
    'high': ['high priority', 'high prio', 'important', 'urgent', 'critical'],
    'med': ['medium priority', 'medium', 'normal priority', 'normal'],
    'low': ['low priority', 'low prio', 'not urgent', 'whenever']
}

FILLER_PREFIXES = ['please', 'can you', 'hey', 'could you', 'I need to', 'I want to', 'let me', 'go ahead and', '']

TEMPLATES = {
    'tier1': [
        {'goal': 'add {title} {due_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'add {title} {due_phrase}, {prio_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'due_phrase': DUE_POOL, 'prio_phrase': PRIO_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'create a task called {title}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'remind me to {title} {due_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'add {title} tagged {tag}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'tag': TAG_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'new task: {title}, {prio_phrase}, {due_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'prio_phrase': PRIO_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': '{title} — add it for {due_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'put {title} on my list', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'add {title} with {prio_phrase} for {due_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'prio_phrase': PRIO_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'schedule {title} for {due_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'create {title} in {tag}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'tag': TAG_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'I need to {title} {due_phrase}, mark it {prio_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL, 'due_phrase': DUE_POOL, 'prio_phrase': PRIO_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'add a {prio_phrase} task: {title}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'prio_phrase': PRIO_POOL, 'title': TITLE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'new {tag} task: {title} {due_phrase}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'tag': TAG_POOL, 'title': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'remember to {title}', 'tier': 1, 'solver': '_solve_direct_add', 'slots': {'title': TITLE_POOL}, 'requires_existing': False, 'min_world_size': 0},
    ],
    'tier2': [
        {'goal': 'how many tasks do I have?', 'tier': 2, 'solver': '_solve_query_count', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'how many {prio_phrase} tasks are due {due_phrase}?', 'tier': 2, 'solver': '_solve_query_count', 'slots': {'prio_phrase': PRIO_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'what tasks are tagged {tag}?', 'tier': 2, 'solver': '_solve_query_list', 'slots': {'tag': TAG_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'list my {due_phrase} tasks', 'tier': 2, 'solver': '_solve_query_list', 'slots': {'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'show me everything due {due_phrase}', 'tier': 2, 'solver': '_solve_query_list', 'slots': {'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'what is on my list for {due_phrase}?', 'tier': 2, 'solver': '_solve_query_list', 'slots': {'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'count my {tag} tasks', 'tier': 2, 'solver': '_solve_query_count', 'slots': {'tag': TAG_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'how many things are due {due_phrase}?', 'tier': 2, 'solver': '_solve_query_count', 'slots': {'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'show all {prio_phrase} tasks', 'tier': 2, 'solver': '_solve_query_list', 'slots': {'prio_phrase': PRIO_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'any tasks for {due_phrase}?', 'tier': 2, 'solver': '_solve_query_list', 'slots': {'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
    ],
    'tier3': [
        {'goal': 'move the {query} task to {due_phrase}', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'change {query} to {prio_phrase}', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL, 'prio_phrase': PRIO_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'mark {query} as done', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'delete the {query} task', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'rename {query} to {new_title}', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL, 'new_title': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'reschedule {query} to {due_phrase}', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'set {query} priority to {prio_val}', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL, 'prio_val': PRIO_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'complete {query}', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'remove {query} from my list', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'tag {query} as {tag}', 'tier': 3, 'solver': '_solve_lookup_then_act', 'slots': {'query': TITLE_POOL, 'tag': TAG_POOL}, 'requires_existing': True, 'min_world_size': 1},
    ],
    'tier4': [
        {'goal': 'mark all {tag} tasks done', 'tier': 4, 'solver': '_solve_bulk', 'slots': {'tag': TAG_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'complete everything due {due_phrase}', 'tier': 4, 'solver': '_solve_bulk', 'slots': {'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'add {title1} for {due1} and {title2} for {due2}', 'tier': 4, 'solver': '_solve_compound', 'slots': {'title1': TITLE_POOL, 'due1': DUE_POOL, 'title2': TITLE_POOL, 'due2': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'finish all {prio_phrase} tasks', 'tier': 4, 'solver': '_solve_bulk', 'slots': {'prio_phrase': PRIO_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'delete all completed tasks', 'tier': 4, 'solver': '_solve_bulk', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'mark all tasks for {due_phrase} as done', 'tier': 4, 'solver': '_solve_bulk', 'slots': {'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'add {title1} and {title2} both due {due_phrase}', 'tier': 4, 'solver': '_solve_compound', 'slots': {'title1': TITLE_POOL, 'title2': TITLE_POOL, 'due_phrase': DUE_POOL}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'complete all {prio_phrase} {tag} tasks', 'tier': 4, 'solver': '_solve_bulk', 'slots': {'prio_phrase': PRIO_POOL, 'tag': TAG_POOL}, 'requires_existing': False, 'min_world_size': 0},
    ],
    'tier5': [
        {'goal': 'what tasks do I have?', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
        {'goal': 'show my tasks', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
        {'goal': 'list everything', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
        {'goal': 'tell me what to do', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
        {'goal': 'all tasks', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
        {'goal': 'show all', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
        {'goal': 'what do I need to do?', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
        {'goal': 'give me my tasks', 'tier': 5, 'solver': '_solve_narrowing', 'slots': {}, 'requires_existing': False, 'min_world_size': 7},
    ],
    'tier6': [
        {'goal': 'mark task 99 as done', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'delete the {query} task', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {'query': ['nonexistent_task_12345']}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'complete {query}', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {'query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 1},
        {'goal': 'edit task 999...', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'move {query} to friday', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {'query': ['never_find_this_one']}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'change priority of 88 to high', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'rename {query} to something else', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {'query': ['missing_task_abc']}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'tag task 77 as work', 'tier': 6, 'solver': '_solve_error_recovery', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
    ],
    'tier7': [
        {'goal': 'finish the {ambiguous_query} task', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
        {'goal': 'delete the {ambiguous_query} task', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
        {'goal': 'mark the {ambiguous_query} done', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
        {'goal': 'move {ambiguous_query} to tomorrow', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
        {'goal': 'change {ambiguous_query} priority to high', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
        {'goal': 'tag {ambiguous_query} as health', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
        {'goal': 'complete {ambiguous_query}', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
        {'goal': 'remove {ambiguous_query} from my list', 'tier': 7, 'solver': '_solve_ambiguity', 'slots': {'ambiguous_query': TITLE_POOL}, 'requires_existing': True, 'min_world_size': 2},
    ],
    'tier8': [
        {'goal': 'book me a flight', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'what is the weather today?', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'write me an essay about dogs', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'translate this to French', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'send an email to John', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'order pizza', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'play some music', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
        {'goal': 'what time is it?', 'tier': 8, 'solver': '_solve_out_of_scope', 'slots': {}, 'requires_existing': False, 'min_world_size': 0},
    ]
}

def add_filler(goal: str, rng: random.Random) -> str:
    prefix = rng.choice(FILLER_PREFIXES)
    if prefix:
        return f"{prefix} {goal}".strip()
    return goal

def random_casing(goal: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.33:
        return goal.lower()
    elif r < 0.66:
        return goal.upper()
    return goal.title()

def inject_typo(goal: str, rng: random.Random) -> str:
    if rng.random() > 0.3 or len(goal) < 4:
        return goal
    idx = rng.randint(0, len(goal) - 2)
    chars = list(goal)
    chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return "".join(chars)

NOISE_TRANSFORMS = [
    lambda g, rng: add_filler(g, rng),
    lambda g, rng: random_casing(g, rng),
    lambda g, rng: inject_typo(g, rng),
]

def sample_goal(tier: str, world, rng: random.Random) -> tuple[str, dict]:
    """Sample a goal from templates for the given tier.
    
    Args:
        tier: tier name like 'tier1'
        world: a TaskStore instance (or None)
        rng: seeded Random instance
    
    Returns:
        (goal_text, template_info_dict)
    """
    templates = TEMPLATES[tier]
    
    # Filter templates by world size requirement
    active_tasks = []
    if world is not None:
        active_tasks = [t for t in world._tasks.values() if t.status != 'deleted']
    world_size = len(active_tasks)
    
    valid = [t for t in templates if world_size >= t.get('min_world_size', 0)]
    if not valid:
        valid = templates
        
    template = rng.choice(valid)
    info = template.copy()
    
    slot_values = {}
    for slot, pool in template.get('slots', {}).items():
        if slot in ['due_phrase', 'due1', 'due2']:
            base = rng.choice(pool)
            phrase = rng.choice(DUE_PHRASES.get(base, [base]))
            slot_values[slot] = phrase
            info[f"{slot}_val"] = base
        elif slot in ['prio_phrase', 'prio_val']:
            base = rng.choice(pool)
            phrase = rng.choice(PRIO_PHRASES.get(base, [base]))
            slot_values[slot] = phrase
            info[f"{slot}_val"] = base
        elif slot == 'ambiguous_query':
            if active_tasks and len(active_tasks) >= 2:
                task = rng.choice(active_tasks)
                # Use first word of title as the ambiguous query
                parts = task.title.split()
                slot_values[slot] = parts[0] if parts else task.title
                info[f"{slot}_val"] = task.title
            else:
                slot_values[slot] = rng.choice(pool)
        elif slot == 'query' and template.get('requires_existing'):
            if active_tasks:
                task = rng.choice(active_tasks)
                parts = task.title.split()
                slot_values[slot] = rng.choice(parts) if parts else task.title
                info[f"{slot}_val"] = task.title
                info["target_id"] = task.id
            else:
                slot_values[slot] = rng.choice(pool)
        else:
            slot_values[slot] = rng.choice(pool)
            
    goal = template['goal'].format(**slot_values)
    
    # Apply noise transforms
    for transform in NOISE_TRANSFORMS:
        goal = transform(goal, rng)
        
    info['goal_text'] = goal
    info['slot_values'] = slot_values
    
    return goal, info

