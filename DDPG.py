import tensorflow as tf
import numpy as np
import vrep
import time
import os
import shutil
import toolregi as tool
tf.reset_default_graph()
np.set_printoptions(suppress=True)

class Robot(object):   
    def __init__(self):       
        self.sim_step = 0.005        
        self.joint_num = 6
        
        self.joint_name = 'UR10_joint'
        self.marker_num = 4
        self.marker_name = 'Marker'
        self.camera_name = 'Camera'
        self.needletip_name = 'NeedleTip'
        self.entry_name = 'ur10tar2'
        self.target_name = 'ur10tar1'
        self.robotend_name = 'tar1'
                
        self.client_ip = '127.0.0.1'        
        self.client_id = -1
        
        self.cameraHandle = -1
        self.joint_handle = np.zeros((self.joint_num,),np.int)
        self.marker_handle = np.zeros((self.marker_num,),np.int)
        self.entry_handle = -1
        self.target_handle = -1              
        self.needletip_handle = -1        
        self.robotend_handle = -1
        
        self.joint_pos = np.zeros((self.joint_num,))
        self.tip_pos = np.zeros((3,))
        self.action = np.zeros((3,))
        
        self.markers = np.zeros((self.marker_num,1,3),)
        self.target_input = np.zeros((3,1),)  
        
        
        self.workspace = [[-235.0, 960.0],
                          [-185.0, 400.0],
                          [160.0, 400.0]]
                                        
        self.tip_target_dis = 0.0
        
    def connection(self):
        #关闭潜在的连接
        vrep.simxFinish(-1)
        #每隔0.2秒检测一次，直到连接上V-rep
        while True:
            self.client_id = vrep.simxStart(self.client_ip,19999,True,True,5000,5)
            if self.client_id > -1:                
                vrep.simxSetFloatSignal(self.client_id, "ConnectFlag", 
                                        19999, vrep.simx_opmode_oneshot)
                break
            else:
                time.sleep(0.2)
                print('Failed connecting to remote API server')            
        print('Connection success!')
                               
    def read_object_handle(self):
        #然后读取Base和Joint的句柄
        for i in range(self.joint_num):
            _, self.joint_handle[i] = vrep.simxGetObjectHandle(self.client_id, 
                                                               self.joint_name+str(i+1), vrep.simx_opmode_blocking)            
        #print(self.joint_handle)
        
        _, self.cameraHandle = vrep.simxGetObjectHandle(self.client_id, 
                                                        self.camera_name, vrep.simx_opmode_blocking)
        
        #再获取四个标记点的句柄
        for i in range(self.marker_num):
            _, self.marker_handle[i] = vrep.simxGetObjectHandle(self.client_id, 
                                                               self.marker_name+str(i), vrep.simx_opmode_blocking)             
        #print(self.marker_handle)
        
        #获取目标路径的两个点的句柄
        _, self.entry_handle = vrep.simxGetObjectHandle(self.client_id, 
                                                        self.entry_name,vrep.simx_opmode_blocking)
        #print(self.entry_handle)              
        _, self.target_handle = vrep.simxGetObjectHandle(self.client_id, 
                                                         self.target_name,vrep.simx_opmode_blocking)
        
        _, self.needletip_handle = vrep.simxGetObjectHandle(self.client_id, 
                                                            self.needletip_name,vrep.simx_opmode_blocking)
        #print(self.target_handle)

        _, self.robotend_handle = vrep.simxGetObjectHandle(self.client_id, 
                                                            self.robotend_name,vrep.simx_opmode_blocking)
        
        print('Handles available!')
    
    def get_state(self):                
        for i in range(self.marker_num):
            _, self.markers[i] = vrep.simxGetObjectPosition(self.client_id, 
                                                            self.marker_handle[i], 
                                                            self.cameraHandle,
                                                            vrep.simx_opmode_streaming)                       
        #tip_current_pos, mid_current_pos = tool.get_current_tip_position(self.markers)                
        self.tip_pos = self.markers[0][0]
        
        for i in range(6):
            _, self.joint_pos[i] = vrep.simxGetJointPosition(self.client_id, 
                             self.joint_handle[i], vrep.simx_opmode_streaming)
        
        vrep.simxSetObjectPosition(self.client_id, self.needletip_handle,
                                   self.cameraHandle, self.tip_pos,
                                   vrep.simx_opmode_oneshot)
        
        #print("JP", self.joint_pos)
                
        _, target_in_worldframe = vrep.simxGetObjectPosition(self.client_id, 
                                                             self.marker_handle[0], 
                                                             -1,
                                                             vrep.simx_opmode_streaming)   
        
        self.tip_target_dis = 1000*np.sqrt(np.sum(np.square(target_in_worldframe-self.target_input)))                                                                   
        return np.hstack([np.ravel(self.joint_pos), np.ravel(self.action),
                          np.ravel(self.tip_pos),np.ravel(self.target_input)])
            
    def reset_target(self):
        if vrep.simxGetConnectionId(self.client_id) == -1:
            self.connection()
        
        for i in range(3):                            
            random_value = np.random.randint(self.workspace[i][0], self.workspace[i][1])
            self.target_input[i] = random_value/1000.0
            
        vrep.simxSetObjectPosition(self.client_id, self.target_handle,
                                   -1, self.target_input,
                                   vrep.simx_opmode_oneshot)
                
    def conduct_action(self, a):        
        self.action = a/1000.0
        if vrep.simxGetConnectionId(self.client_id) == -1:
            self.connection()
        
        #print("action2", a)
        lasttime = time.time()
        while True:           
            for i in range(3):                                
                vrep.simxSetFloatSignal(self.client_id, "myTestValue"+str(i), 
                                        self.action[i], vrep.simx_opmode_oneshot)           
            currtime = time.time()
            if currtime-lasttime > 0.1:
                break
    
    def get_reward(self):
        
        if self.tip_target_dis > 2:
            r = -self.tip_target_dis/100.0
        else:
            r = 20
        return r

np.random.seed(1)
tf.set_random_seed(1)
MAX_EPISODES = 2000
MAX_EP_STEPS = 200  # hong modified
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.98  # 0.98  # reward discount  # hong modified
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
VAR_MIN = 0.1
RENDER = True
LOAD = False
MODE = ['easy', 'hard']
n_model = 1

STATE_DIM = 15
ACTION_DIM = 3
ACTION_BOUND = [-1, 1]

Workspace = [[-235.0, 960.0],
             [-285.0, 600.0],
             [160.0, 400.0]]

print('STATE_DIM')
print(STATE_DIM)

print('ACTION_DIM')
print(ACTION_DIM)

print('ACTION_BOUND')
print(ACTION_BOUND)

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a_, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.001), name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.001), name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.001), name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='a', trainable=trainable)                
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1

    def choose_action(self, s, noise=None):
        s = s[np.newaxis, :]    # single state
        
        c_a = self.sess.run(self.a, feed_dict={S: s})[0]  # Shape(3), in [-1,1]
        print(c_a)
        
        for i in range(3):  # apply transform here
            if not noise is None:
                c_a[i] = np.clip(c_a[i]+noise[i], -1,1)
            c_a[i] = c_a[i]*(Workspace[i][1]-Workspace[i][0])/2+(Workspace[i][1]+Workspace[i][0])/2
                    
        #print("action0", c_a)
        
        return c_a  # single action
    
    

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=tf.constant_initializer(0.01), trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.01), name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.constant_initializer(0.01), name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2), bias_initializer=tf.constant_initializer(0.01), trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
   
sess = tf.Session()
# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)
saver = tf.train.Saver()
path = './'+MODE[n_model]

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

ur_robot = Robot()
ur_robot.connection()
ur_robot.read_object_handle()

class OuProcess():
    def __init__(self,x=.5*np.random.randn(), sigma=0.3, mu=0, theta=1, dt=1e-2):
        # params follows:
        # https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
        self.x = x
        self.sigma=sigma
        self.mu=mu
        self.theta=theta
        self.dt = dt
        self.sqrtdt = np.sqrt(dt)

    def fetch_next(self):
        # self.x = self.x + self.dt * (-(self.x - self.mu) / self.tau) + self.sigma_bis * self.sqrtdt * np.random.randn()
        self.x = self.x + self.theta*(self.mu - self.x)*self.dt + self.sigma*self.sqrtdt*np.random.randn()
        return self.x
    
    
def train():    
    var = 3.  # control exploration
    for ep in range(MAX_EPISODES):        
        ur_robot.reset_target()
        s = ur_robot.get_state()
        ep_reward = 0

        done = False
        cur_time = time.time()
        # (x=.5*np.random.randn(), sigma=0.3, mu=0, theta=1, dt=1e-2)
        ou_processes = [OuProcess(x=2*np.random.randn(), sigma=0.003, mu=0, theta=1.289, dt=1.5) for _ in range(3)] # 3 processes
        for t in range(MAX_EP_STEPS):
            # Added exploration noise
            noise = [p.fetch_next() for p in ou_processes]  # noises of 3-dims
            # noise=None  # disable
            print(ep, t,':',noise)
            a = actor.choose_action(s, noise)
                                
            #a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration                        
            #print("action1",a)  
            ur_robot.conduct_action(a)            
            s_= ur_robot.get_state()
            r = ur_robot.get_reward()
            if r > 10:
                done = True
            
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result, 
                      '| step: %i' % t,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      '| Time: %.2f' % (time.time()-cur_time)
                      )
                break
                    
    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./'+MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)

def eval():
    ur_robot.reset_target()
    s = ur_robot.get_state()
    while True:       
        a = actor.choose_action(s)                               
        ur_robot.conduct_action(a)            
        s_= ur_robot.get_state()
        s = s_
   
if __name__ == '__main__':
    
    if LOAD:
        eval()
    else:
        train()
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
