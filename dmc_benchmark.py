DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'ant',
    'point_mass_maze'
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

CHEETAH_TASKS = [
    'cheetah_walk',
    'cheetah_run',
    'cheetah_flip',
    'cheetah_walk_backward',
    'cheetah_run_backward',
    'cheetah_flip_backward',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

ANT_TASKS = [
    'forward',
    'backward',
    'left',
    'right',
    'goal',
    'motion',
]

HALFCHEETAH_TASKS = [
    'default',
    'target_velocity',
    'run_back',
]

POINT_MASS_MAZE_TASKS = [
    'point_mass_maze_reach_top_left',
    'point_mass_maze_reach_top_right',
    'point_mass_maze_reach_bottom_left',
    'point_mass_maze_reach_bottom_right',
]
               
TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + ANT_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_walk',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'ant': 'forward',
    'half_cheetah': 'default',
    'cheetah': 'cheetah_walk',
    'hopper': 'hopper_hop',
    'humanoid': 'humanoid_run',
    # 'mw': 'mw_pick-place',
    # 'mw': 'mw_button-press-topdown',
    'mw': 'mw_faucet-open',
    'mw1': 'mw1_pick-place',
    'point_mass_maze': 'point_mass_maze_reach_top_left'
    # 'mw': 'mw_pick-place',
    # 'mw1': 'mw1_pick-place'
}
