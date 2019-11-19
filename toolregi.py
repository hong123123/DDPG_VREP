# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 08:55:08 2019

@author: Lenovo
"""
import numpy as np
import math

def calculateMiddleRT(pointset):
    
#    print('pointset')
#    print(pointset)
    
    #计算点集的自身旋转平移矩阵
    TMat = np.array(pointset[0])
    TMat = -TMat
            
    Vx = np.array([1.,0.,0.])
    
    Vy = np.array([0.,1.,0.])
    
    Vz = np.array([0.,0.,1.])
    
    np.transpose(Vx)
    np.transpose(Vy)
    np.transpose(Vz)
    
    P0 = np.array([[0.,0.,0.],
                   [0.,0.,0.],
                   [0.,0.,0.]])
        
    for i in range(3):
        P0[i] = TMat + pointset[i]
    
    Vabxoy = np.array([0.,0.,0.])
    np.transpose(Vabxoy)
    
    Vabxoy[0] = P0[1][0]
    Vabxoy[1] = P0[1][1]
    Vabxoy[2] = 0.
          
    Vabxoy_dot_Vx = np.dot(Vabxoy,Vx)
    Vabxoy_dot_Vabxoy = np.dot(Vabxoy,Vabxoy)
    Vx_dot_Vx = np.dot(Vx,Vx)
    
    mid_val_z = Vabxoy_dot_Vx/math.sqrt(Vabxoy_dot_Vabxoy*Vx_dot_Vx)
        
    angle_rotate_z = math.acos(mid_val_z)
             
    if P0[1][1] > 0:        
        angle_rotate_z = -angle_rotate_z
        
    rotate_data_z = np.array([[math.cos(angle_rotate_z), -math.sin(angle_rotate_z), 0],
                      [math.sin(angle_rotate_z), math.cos(angle_rotate_z), 0],
                      [0,0,1]])
           
    P0 = Matrix_multi(rotate_data_z, P0.T)
   
    P0 = P0.T
 
    Vabxoz = P0[1,:]

    angle_rotate_y = math.acos(np.dot(Vabxoz,Vx)/math.sqrt(np.dot(Vabxoz,Vabxoz)*np.dot(Vx,Vx)))
    if P0[1,2] < 0:
        angle_rotate_y = -angle_rotate_y
    
    rotate_data_y = np.array([[math.cos(angle_rotate_y), 0., math.sin(angle_rotate_y)],
                      [0.,1.,0.],
                      [-math.sin(angle_rotate_y), 0., math.cos(angle_rotate_y)]])
    
    P0 = Matrix_multi(rotate_data_y, P0.T)    
    P0 = P0.T
    
    Vacyoz = np.array([0.,0.,0.])
    np.transpose(Vacyoz)
    
    Vacyoz[1] = P0[2,1]
    Vacyoz[2] = P0[2,2]
    Vacyoz[0] = 0
    
    np.transpose(Vacyoz)
        
    Vacyoz_dot_Vy = np.dot(Vacyoz,Vy)
    Vacyoz_dot_Vacyoz = np.dot(Vacyoz,Vacyoz)
    Vy_dot_Vy = np.dot(Vy,Vy)
            
    mid_val_x = Vacyoz_dot_Vy/math.sqrt(Vacyoz_dot_Vacyoz*Vy_dot_Vy)
    
    angle_rotate_x = math.acos(mid_val_x)
    
    if P0[2,2] > 0:
        angle_rotate_x = -angle_rotate_x
                    
    rotate_data_x = np.array([[1,0,0], 
                     [0, math.cos(angle_rotate_x), -math.sin(angle_rotate_x)], 
                     [0, math.sin(angle_rotate_x), math.cos(angle_rotate_x)]])
        
    ROutput = Matrix_multi(Matrix_multi(rotate_data_x,rotate_data_y),rotate_data_z)
    TOutput = Matrix_multi(ROutput,np.transpose(TMat))
        
    return ROutput,TOutput

def Matrix_multi(m1,m2):
    
    #首先建立一个值都是0的矩阵,矩阵形状是矩阵1的行数和矩阵2的列数组成
    results = np.zeros((m1.shape[0],m2.shape[1]))
    #判断矩阵1的列和矩阵2的行数是否相同,如果不相同,则两个矩阵无法相乘,就直接返回
    if m1.shape[1] != m2.shape[0]:
        return
    #首先遍历矩阵1的行
    for i in range(m1.shape[0]):
    	#这是遍历矩阵2的列
        for j in range(m2.shape[1]):
            sum = 0
            #这里遍历矩阵1的列和遍历矩阵2的行都可以,因为他们是相同的
            for k in range(m2.shape[0]):
            	#把对应位置相乘并相加后得到的值放入指定位置
                sum += (m1[i][k] * m2[k][j])
                results[i][j] = sum
                
    return results


def Transform_Pointsets(pointset1,pointset2):
    R1mat,T1mat = calculateMiddleRT(pointset1)    
    R2mat,T2mat = calculateMiddleRT(pointset2)    
    Ro = Matrix_multi(np.linalg.inv(R1mat),R2mat)
    To = Matrix_multi(np.linalg.inv(R1mat),(T2mat-T1mat))
    
    return Ro,To
    
    
def get_current_tip_position(marker_config):    
    toolregi_result_tip = np.array([[-188.22, -41.106, -40.477]])
    toolregi_result_mid = np.array([[-90.407, -18.688, -40.355]])
    toolregi_result_tip = toolregi_result_tip.T
    toolregi_result_mid = toolregi_result_mid.T       
    pointset2 = np.zeros((3,1,3),)    
    pointset2[0] = [48.56,57.95,0]
    pointset2[1] = [0,0,0]
    pointset2[2] = [114.77,0,0]    
    distance_config = [[101,87,75],
                       [75,114,150],
                       [45,114,87],
                       [101,45,150]]    
    #获取工具到光学的旋转平移矩阵        
    input_pointset = np.zeros((3,1,3),)        
    for i in range(4):
        point0 = marker_config[i]
        dislist = np.zeros((3,))       
        dis_num = 0
        for j in range(4):            
            if(j!=i):
                point1 = marker_config[j]                
                dis = np.sqrt(np.sum(np.square(point0-point1)))                
                dis = 1000*dis
                dislist[dis_num] = math.floor(dis)
                dis_num = dis_num + 1
        dis_num = 0                 
        _dislist = list(dislist)      
        for k in range(4):
            dis_config = distance_config[k]           
            _dis_config = list(dis_config)            
            sumlist = list(set(_dis_config).union(set(_dislist))) 
            list_len = len(sumlist)            
            if list_len==3:
                if k!=3:
                    input_pointset[k] = point0                                    
    input_pointset = input_pointset*1000          
    tool2ots_r, tool2ots_t = Transform_Pointsets(input_pointset, pointset2)            
    tooltip_in_cameraframe = Matrix_multi(tool2ots_r,toolregi_result_tip)+tool2ots_t
    toolmid_in_cameraframe = Matrix_multi(tool2ots_r,toolregi_result_mid)+tool2ots_t    
    return tooltip_in_cameraframe,toolmid_in_cameraframe
      
    
    
    
    
    
    
    
    