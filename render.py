from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import numpy as np
import pandas as pd

width = 512
height = 512

# width = 500
# height = 300
image = Image(width, height, Color(255, 255, 255, 255))

dataset = pd.read_csv('data/IMUData.csv')

# Init z-buffer
zBuffer = [-float('inf')] * width * height

#Set projection type
# projection = 'orthographic'
projection = 'perspective'

# Load the model
model = Model('data/headset.obj')
model.normalizeGeometry()

### HELPER FUNCTIONS ###

def transformModel(transformationMatrix, model):
	''' Apply transformation matrix to model vertices (rotation, translation, scaling)'''

	print("transformationMatrix:", transformationMatrix)

	# move model to the origin
	origin = np.array([0, 0, 0])
	for i in range(len(model.vertices)):
		model.vertices[i] = model.vertices[i] - origin

	# Apply transformation matrix
	for i in range(len(model.vertices)):
		point = np.array([model.vertices[i].x, model.vertices[i].y, model.vertices[i].z,1])
		transformed_point = np.dot(transformationMatrix, point)
		model.vertices[i] = Vector(transformed_point[0], transformed_point[1], transformed_point[2])

	# move model back to original position
	for i in range(len(model.vertices)):
		model.vertices[i] = model.vertices[i] + origin

	return model

def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)

	return screenX, screenY

def viewMatrix(eye_pos, target, up):
	up = up/np.linalg.norm(up)

	z = (eye_pos - target)/np.linalg.norm(eye_pos - target)


	x = np.cross(up, z)/np.linalg.norm(np.cross(up, z))

	y = np.cross(z, x)

	viewMat= np.matrix([
		[x[0],x[1],x[2],-((x[0]*eye_pos[0]) + (x[1]*eye_pos[1]) + (x[2]*eye_pos[2]))],
		[y[0],y[1],y[2],-((y[0]*eye_pos[0]) + (y[1]*eye_pos[1]) + (y[2]*eye_pos[2]))],
		[z[0],z[1],z[2],-((z[0]*eye_pos[0]) + (z[1]*eye_pos[1]) + (z[2]*eye_pos[2]))],
		[0,0,0,1]])
	
	return viewMat



def getPerspectiveProjection(x, y, z, d=200):
	# Assuming eye/camera at (0, 0, 0) 
	ar = width / height # Aspect ratio
	n = 400 # Near plane
	f = 750 # Far plane
	r = width / 2 # Right
	l = -r # Left
	t = height / 2 # Top
	b = -t # Bottom
	
	T_st = np.array([[((2*n)/(r-l)),0,((r+l)/(r-l)),0],
					 [0,((2*n)/(t-b)),((t+b)/(t-b)),0],
					 [0,0,-((f+n)/(f-n)),-((2*f*n)/(f-n))],
					 [0,0,-1,0]])
	
	T_vp = np.array([[width/2, 0, 0, (width-1)/2],
					 [0, height/2, 0, (height-1)/2],
					 [0, 0, 1/2, 0],
					 [0, 0, 0, 1]])

	T_v = viewMatrix(np.array([0, 0, -2]), np.array([0, 1, 1]), np.array([0, 0, -1]))

	clip = T_st * T_v * np.array([[x], [y], [z], [1]])

	new_x = clip[0,0]/clip[3,0]
	new_y = clip[1,0]/clip[3,0]
	new_z = clip[2,0]/clip[3,0]

	screenX = int((new_x+1.0)*(width/2.0))
	screenY = int((new_y+1.0)*(height/2.0))
	screenZ = int((new_z+1.0)*(1.0/2.0))

	return screenX, screenY, screenZ

def translateMatrix(a,b,c):
	return np.array([[1, 0, 0, a],
					 [0, 1, 0, b],
					 [0, 0, 1, c],
					 [0, 0, 0, 1]])


def rotateMatrix(angle, axis):
	angle = np.radians(angle)
	
	cos = np.cos(angle)
	sin = np.sin(angle)

	if axis == 'x':
		return np.array([[1, 0, 0, 0],
						 [0, cos, -sin, 0],
						 [0, sin, cos, 0],
						 [0, 0, 0, 1]])
	elif axis == 'y':
		return np.array([[cos, 0, sin, 0],
						 [0, 1, 0, 0],
						 [-sin, 0, cos, 0],
						 [0, 0, 0, 1]])
	elif axis == 'z':
		return np.array([[cos, -sin, 0, 0],
						 [sin, cos, 0, 0],
						 [0, 0, 1, 0],
						 [0, 0, 0, 1]])
	

def scaleMatrix(a, b, c):
	return np.array([[a, 0, 0, 0],
					 [0, b, 0, 0],
					 [0, 0, c, 0],
					 [0, 0, 0, 1]])



def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])

# Calculate face normals
faceNormals = {}
for face in model.faces:
	p0, p1, p2 = [model.vertices[i] for i in face]
	faceNormal = (p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceNormals:
			faceNormals[i] = []

		faceNormals[i].append(faceNormal)

# Calculate vertex normals
vertexNormals = []
for vertIndex in range(len(model.vertices)):
	vertNorm = getVertexNormal(vertIndex, faceNormals)
	vertexNormals.append(vertNorm)

def updateDataset(dataset):
	dataset[' gyroscope.X'] = np.radians(dataset[' gyroscope.X'])
	dataset[' gyroscope.Y'] = np.radians(dataset[' gyroscope.Y'])
	dataset[' gyroscope.Z'] = np.radians(dataset[' gyroscope.Z'])


	return dataset

def eulerToQuaternions(roll, pitch, yaw):
	''' Convert Euler angle readings (radians) to quaternions

		roll: Rotation around x-axis
		pitch: Rotation around y-axis
		yaw: Rotation around z-axis (Z IS UP!)
	'''

	q_0 = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	q_1 = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
	q_2 = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
	q_3 = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

	return q_0, q_1, q_2, q_3

def quaternionsToEuler(q_1, q_2, q_3, q_4):
	''' Convert quaternions to Euler angles (radians)'''

	roll = np.arctan2(2*(q_1*q_2 + q_3*q_4), 1 - 2*(q_2**2 + q_3**2))
	pitch = np.arcsin(2*(q_1*q_3 - q_4*q_2))
	yaw = np.arctan2(2*(q_1*q_4 + q_2*q_3), 1 - 2*(q_3**2 + q_4**2))

	return roll, pitch, yaw

def conjugateOfQuaternion(q_0, q_1, q_2, q_3):
	''' Calculate the conjugate of a quaternion (inverse rotation)'''

	return q_0, -q_1, -q_2, -q_3

def multiplyQuaternions(q_0, q_1, q_2, q_3, p_0, p_1, p_2, p_3):
	''' Multiply two quaternions (Hamilton product)'''

	r_0 = q_0*p_0 - q_1*p_1 - q_2*p_2 - q_3*p_3
	r_1 = q_0*p_1 + q_1*p_0 + q_2*p_3 - q_3*p_2
	r_2 = q_0*p_2 - q_1*p_3 + q_2*p_0 + q_3*p_1
	r_3 = q_0*p_3 + q_1*p_2 - q_2*p_1 + q_3*p_0

	return r_0, r_1, r_2, r_3

def quaternionToMatrix(q_0, q_1, q_2, q_3):
	''' Convert quaternions to rotation matrix'''

	R = np.array([[1 - 2*q_2**2 - 2*q_3**2, 2*q_1*q_2 - 2*q_0*q_3, 2*q_1*q_3 + 2*q_0*q_2,0],
				  [2*q_1*q_2 + 2*q_0*q_3, 1 - 2*q_1**2 - 2*q_3**2, 2*q_2*q_3 - 2*q_0*q_1,0],
				  [2*q_1*q_3 - 2*q_0*q_2, 2*q_2*q_3 + 2*q_0*q_1, 1 - 2*q_1**2 - 2*q_2**2,0],
				  [0,0,0,1]])

	return R

def axisToQuaternion(axis, angle):
	''' Convert axis-angle representation to quaternions'''

	axis = axis/np.linalg.norm(axis)
	q_0 = np.cos(angle/2)
	q_1 = np.sin(angle/2)*axis[0]
	q_2 = np.sin(angle/2)*axis[1]
	q_3 = np.sin(angle/2)*axis[2]

	return q_0, q_1, q_2, q_3

def quaternionToAxis(q_0, q_1, q_2, q_3):
	''' Convert quaternions to axis-angle representation'''
	# if q_0 == 1:
	# 	angle = 0
	# 	x = 1
	# 	y = 0
	# 	z = 0
	# else:
	angle = 2*np.arccos(q_0)
	axis = np.array([q_1, q_2, q_3])/np.sin(angle/2)

	return angle, axis

def deadReckoning(k,gyro, prev):
	''' Perform dead reckoning to estimate the orientation of the device at time k '''
	# delta_t = dataset['time'][k] - dataset['time'][k-1]

	omega = np.linalg.norm(gyro)*(100/256)

	axis = gyro/np.linalg.norm(gyro)
	q_gyro = axisToQuaternion(axis, omega)
	
	current_orientation = multiplyQuaternions(prev[0], prev[1], prev[2], prev[3], q_gyro[0], q_gyro[1], q_gyro[2], q_gyro[3])

	return current_orientation


def transformAcceleration(dataset, q_0, q_1, q_2, q_3):
	''' Transform acceleration measurements from the device frame to the global frame '''

	a_0 = 0
	a_1 = dataset[' accelerometer.X']
	a_2 = dataset[' accelerometer.Y']
	a_3 = dataset[' accelerometer.Z']

	q_conj = conjugateOfQuaternion(q_0, q_1, q_2, q_3)
	
	transformed_accel = multiplyQuaternions(q_0, q_1, q_2, q_3, a_0, a_1, a_2, a_3)

	transformed_accel = multiplyQuaternions(transformed_accel[0], transformed_accel[1], transformed_accel[2], transformed_accel[3], q_conj[0], q_conj[1], q_conj[2], q_conj[3])

	return transformed_accel[1], transformed_accel[2], transformed_accel[3]

def calculateTiltError(k,accelX, accelY, accelZ):
	''' Calculate the tilt error of the device
	
		accelX: transformed acceleration in x-direction	
		accelY: transformed acceleration in y-direction
		accelZ: transformed acceleration in z-direction
	'''
	tilt_axis = np.array([accelZ[k], 0, -accelX[k]])
	
	phi = np.arccos(np.dot([accelX[k], accelY[k], accelZ[k]], np.array([0, 0, 1]))/np.linalg.norm([accelX[k], accelY[k], accelZ[k]])*1)

	return tilt_axis, phi

def complementaryFilter(k, dataset, current_orientation, alpha):
	''' Perform sensor fusion using the complementary filter '''
	
	gyro = np.array([dataset[' gyroscope.X'][k], dataset[' gyroscope.Y'][k], dataset[' gyroscope.Z'][k]])
	current_orientation = deadReckoning(k,gyro, current_orientation)

	transformed_accelX, transformed_accelY, transformed_accelZ = transformAcceleration(dataset, current_orientation[0], current_orientation[1], current_orientation[2], current_orientation[3])

	tilt_axis, phi = calculateTiltError(k,transformed_accelX, transformed_accelY, transformed_accelZ)
	
	q_tilt = axisToQuaternion(tilt_axis, alpha*phi)

	# q_fusion = multiplyQuaternions(current_orientation[0], current_orientation[1], current_orientation[2], current_orientation[3], q_tilt[0], q_tilt[1], q_tilt[2], q_tilt[3])
	q_fusion = multiplyQuaternions(q_tilt[0], q_tilt[1], q_tilt[2], q_tilt[3], current_orientation[0], current_orientation[1], current_orientation[2], current_orientation[3])
	
	return q_fusion

def transformMag(dataset, q_0, q_1, q_2, q_3):
	''' Transform magnetometer measurements from the device frame to the global frame '''

	m_0 = 0
	m_1 = dataset[' magnetometer.X']
	m_2 = dataset[' magnetometer.Y']
	m_3 = dataset[' magnetometer.Z ']

	q_conj = conjugateOfQuaternion(q_0, q_1, q_2, q_3)
	
	transformed_mag = multiplyQuaternions(q_0, q_1, q_2, q_3, m_0, m_1, m_2, m_3)

	transformed_mag = multiplyQuaternions(transformed_mag[0], transformed_mag[1], transformed_mag[2], transformed_mag[3], q_conj[0], q_conj[1], q_conj[2], q_conj[3])

	return transformed_mag[1], transformed_mag[2], transformed_mag[3]

def angularDifference(ref, actual):
	omega = np.arctan2(actual[0], actual[2])

	omega_r = np.arctan2(ref[0], ref[2])

	error = abs(omega - omega_r)

	return omega, omega_r, error

def complementaryFilter2(k, dataset, current_orientation, alpha_2 = 0.1):
	global errors
	global actual
	transformed_magX, transformed_magY, transformed_magZ = transformMag(dataset, current_orientation[0], current_orientation[1], current_orientation[2], current_orientation[3])
	ref, actual = actual, [transformed_magX[k], transformed_magY[k], transformed_magZ[k]]
	omege, omera_r, error = angularDifference(ref, actual)
	errors.append(error)

	q_mag = axisToQuaternion([0, 1, 0], -alpha_2*error)
	current_orientation = multiplyQuaternions(q_mag[0], q_mag[1], q_mag[2], q_mag[3], current_orientation[0], current_orientation[1], current_orientation[2], current_orientation[3])

	return current_orientation

### OUTPUT FUNCTION
current_orientation = np.array([1, 0, 0, 0])
actual = [0,1,0]
errors = []

def outputData():
	global model
	global current_orientation
	
	lightDir = Vector(0, 0, -1)

	for k in range(1,len(dataset)):
		print("k:", k)
		# Perform sensor fusion using the complementary filter
		# q_fusion = deadReckoning(k, np.array([dataset[' gyroscope.X'][k], dataset[' gyroscope.Y'][k], dataset[' gyroscope.Z'][k]]), current_orientation)
		q_fusion = complementaryFilter(k, dataset, current_orientation, alpha=0.1)
		q_fusion = complementaryFilter2(k, dataset, q_fusion, alpha_2 = 0.2)

		q_fusion_inverse = conjugateOfQuaternion(q_fusion[0], q_fusion[1], q_fusion[2], q_fusion[3])
		current_orientation = q_fusion
		
		image = Image(width, height, Color(255, 255, 255, 255))
		zBuffer = [-float('inf')] * width * height
		for face in model.faces:
			p0, p1, p2 = [model.vertices[i] for i in face]
			n0, n1, n2 = [vertexNormals[i] for i in face]

			cull = False

			transformedPoints = []
			for p, n in zip([p0, p1, p2], [n0, n1, n2]):

				point = np.array([0, p.x, p.y, p.z])
				
				# Rotate the point
				first_mult = multiplyQuaternions(q_fusion[0], q_fusion[1], q_fusion[2], q_fusion[3], 0, point[1], point[2], point[3])
				second_mult = multiplyQuaternions(first_mult[0], first_mult[1], first_mult[2], first_mult[3], q_fusion_inverse[0], q_fusion_inverse[1], q_fusion_inverse[2], q_fusion_inverse[3])

				_, p = quaternionToAxis(second_mult[0], second_mult[1], second_mult[2], second_mult[3])

				p = Point(p[0], p[1], p[2], Color(255, 255, 255, 255))

				intensity = n * lightDir
		
				if projection == 'orthographic':
					screenX, screenY = getOrthographicProjection(p.x, p.y, p.z)
					screenZ = 0
					intensity = Vector(screenX, screenY, screenZ).normalize() * lightDir

				elif projection == 'perspective':    
					screenX, screenY, screenZ = getPerspectiveProjection(p.x, p.y, p.z)
					intensity = Vector(screenX, screenY, screenZ).normalize() * lightDir

				transformedPoints.append(Point(screenX, screenY, screenZ, Color(intensity*255, intensity*255, intensity*255, 255)))
			
			if intensity < 0:
					cull = True
		
			if not cull:	
				Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)

		display_pause(image)

# Display the image
def display_pause(image):
	image.saveAsPNG("image.png")
	plt.ion() 
	im = plt.imread('image.png')
	plt.imshow(im)
	plt.pause(0.001)
	# plt.waitforbuttonpress()
