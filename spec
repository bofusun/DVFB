Help on AntEnv in module mujoco_envs.ant_env object:

class AAnnttEEnnvv(mujoco_envs.mujoco_utils.MujocoTrait, gym.envs.mujoco.mujoco_env.MujocoEnv, gym.utils.ezpickle.EzPickle)
 |  AntEnv(task='motion', goal=None, expose_obs_idxs=None, expose_all_qpos=True, expose_body_coms=None, expose_body_comvels=None, expose_foot_sensors=False, use_alt_path=False, model_path=None, fixed_initial_state=False, done_allowing_step_unit=None, original_env=False, render_hw=100)
 |  
 |  Superclass for all MuJoCo environments.
 |  
 |  Method resolution order:
 |      AntEnv
 |      mujoco_envs.mujoco_utils.MujocoTrait
 |      gym.envs.mujoco.mujoco_env.MujocoEnv
 |      gym.core.Env
 |      gym.utils.ezpickle.EzPickle
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ____iinniitt____(self, task='motion', goal=None, expose_obs_idxs=None, expose_all_qpos=True, expose_body_coms=None, expose_body_comvels=None, expose_foot_sensors=False, use_alt_path=False, model_path=None, fixed_initial_state=False, done_allowing_step_unit=None, original_env=False, render_hw=100)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  ccaallcc__eevvaall__mmeettrriiccss(self, trajectories, is_option_trajectories)
 |  
 |  ccoommppuuttee__rreewwaarrdd(self, **kwargs)
 |  
 |  rreesseett__mmooddeell(self)
 |      Reset the robot degrees of freedom (qpos and qvel).
 |      Implement this in each subclass.
 |  
 |  sstteepp(self, a, render=False)
 |      Run one timestep of the environment's dynamics. When end of
 |      episode is reached, you are responsible for calling `reset()`
 |      to reset this environment's state.
 |      
 |      Accepts an action and returns a tuple (observation, reward, done, info).
 |      
 |      Args:
 |          action (object): an action provided by the agent
 |      
 |      Returns:
 |          observation (object): agent's observation of the current environment
 |          reward (float) : amount of reward returned after previous action
 |          done (bool): whether the episode has ended, in which case further step() calls will return undefined results
 |          info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
 |  
 |  vviieewweerr__sseettuupp(self)
 |      This method is called when the viewer is initialized.
 |      Optionally implement this method, if you need to tinker with camera position
 |      and so forth.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties defined here:
 |  
 |  bbooddyy__ccoomm__iinnddiicceess
 |  
 |  bbooddyy__ccoommvveell__iinnddiicceess
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from mujoco_envs.mujoco_utils.MujocoTrait:
 |  
 |  pplloott__ttrraajjeeccttoorriieess(self, trajectories, colors, plot_axis, ax)
 |  
 |  pplloott__ttrraajjeeccttoorryy(self, trajectory, color, ax)
 |  
 |  rreennddeerr(self, mode='human', width=100, height=100, camera_id=None, camera_name=None)
 |  
 |  rreennddeerr__ttrraajjeeccttoorriieess(self, trajectories, colors, plot_axis, ax)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from mujoco_envs.mujoco_utils.MujocoTrait:
 |  
 |  ____ddiicctt____
 |      dictionary for instance variables (if defined)
 |  
 |  ____wweeaakkrreeff____
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from gym.envs.mujoco.mujoco_env.MujocoEnv:
 |  
 |  cclloossee(self)
 |      Override close in your subclass to perform any necessary cleanup.
 |      
 |      Environments will automatically close() themselves when
 |      garbage collected or when the program exits.
 |  
 |  ddoo__ssiimmuullaattiioonn(self, ctrl, n_frames)
 |  
 |  ggeett__bbooddyy__ccoomm(self, body_name)
 |  
 |  rreesseett(self)
 |      Resets the state of the environment and returns an initial observation.
 |      
 |      Returns: 
 |          observation (object): the initial observation.
 |  
 |  sseeeedd(self, seed=None)
 |      Sets the seed for this env's random number generator(s).
 |      
 |      Note:
 |          Some environments use multiple pseudorandom number generators.
 |          We want to capture all such seeds used in order to ensure that
 |          there aren't accidental correlations between multiple generators.
 |      
 |      Returns:
 |          list<bigint>: Returns the list of seeds used in this env's random
 |            number generators. The first value in the list should be the
 |            "main" seed, or the value which a reproducer should pass to
 |            'seed'. Often, the main seed equals the provided 'seed', but
 |            this won't be true if seed=None, for example.
 |  
 |  sseett__ssttaattee(self, qpos, qvel)
 |  
 |  ssttaattee__vveeccttoorr(self)
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from gym.envs.mujoco.mujoco_env.MujocoEnv:
 |  
 |  ddtt
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from gym.core.Env:
 |  
 |  ____eenntteerr____(self)
 |  
 |  ____eexxiitt____(self, *args)
 |  
 |  ____ssttrr____(self)
 |      Return str(self).
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from gym.core.Env:
 |  
 |  uunnwwrraappppeedd
 |      Completely unwrap this env.
 |      
 |      Returns:
 |          gym.Env: The base non-wrapped gym.Env instance
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from gym.core.Env:
 |  
 |  aaccttiioonn__ssppaaccee = None
 |  
 |  mmeettaaddaattaa = {'render.modes': []}
 |  
 |  oobbsseerrvvaattiioonn__ssppaaccee = None
 |  
 |  rreewwaarrdd__rraannggee = (-inf, inf)
 |  
 |  ssppeecc = None
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from gym.utils.ezpickle.EzPickle:
 |  
 |  ____ggeettssttaattee____(self)
 |  
 |  ____sseettssttaattee____(self, d)
