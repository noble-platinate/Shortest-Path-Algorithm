import cv2 as cv
import numpy as np
import math
from queue import PriorityQueue
import time

#ASSIGN RGB VALUES TO SPECIFIC VARIABLES
START_COLOR = (113,204,45)
END_COLOR = (60,76,231)
OBSTACLES_COLOR = (255,255,255)
NAVIGABLE_PATH_COLOR = (0,0,0)

PURPLE = (128, 0, 128)
RED = (0, 100, 250)
GREEN = (100, 100, 0)

#READ THE IMAGE OF THE MAZE
img=cv.imread("maze.png",1)
copy=img
HEIGHT = img.shape[0]
WIDTH = img.shape[1]

#FIND THE COORDINATES OF THE START AND END POINT
for i in range(HEIGHT):
    for j in range(WIDTH):
        if img[i][j][2]==START_COLOR[2]:
            start=(i,j)
        if img[i][j][2]==END_COLOR[2]:
            end=(i,j)

#CALCULATE THE DISTANCE BETWEEN 2 POINTS 
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    return abs(x1 - x2) + abs(y1 - y2)                              #MANHATTAN DISTANCE
    #return max(abs(x1 - x2),abs(y1 - y2))                          #DIAGONAL DISTANCE
    #return math.dist([x1,y1],[x2,y2])                              #EUCLIDIAN DISTANCE
    #return 5*math.dist([x1,y1],[x2,y2])                            #NON-ADMISSIBLE DISTANCE
    #return (abs(x1 - x2) + abs(y1 - y2))/2                         #ADMISSIBLE DISTANCE
    #return 0                                                       #DIJKSTRA DISTANCE

#LIST THE POSSIBLE NEIGHBORS OF A NODE
def is_defined(current,ROWS,COLUMNS,img):
    
    neighbors=[]

    #CHECK THE NON-DIAGONAL NEIGHBORS WHETHER OBSTACLE OR NOT
    if (current[0]<ROWS-1) and not (img[current[0]+1][current[1]][0]==255):
        neighbors.append((current[0]+1,current[1]))
        
    if (current[0]>0) and not (img[current[0]-1][current[1]][0]==255):
        neighbors.append((current[0]-1,current[1]))

    if (current[1]<COLUMNS-1) and not (img[current[0]][current[1]+1][0]==255):
        neighbors.append((current[0],current[1]+1))

    if (current[1]>0) and not (img[current[0]][current[1]-1][0]==255):
        neighbors.append((current[0],current[1]-1))

    #CHECK THE DIAGONAL NEIGHBORS WHETHER OBSTACLE OR NOT
    #COMMENT LINE 59-69 FOR NON_DIAG MOVEMENT DISALLOWED A* ALGORITHM
    if (current[0]<ROWS-1) and (current[1]<COLUMNS-1) and not (img[current[0]+1][current[1]+1][0]==255):
        neighbors.append((current[0]+1,current[1]+1))

    if (current[0]>0) and (current[1]<COLUMNS-1) and not (img[current[0]-1][current[1]+1][0]==255):
        neighbors.append((current[0]-1,current[1]+1))

    if (current[1]>0) and (current[0]<ROWS-1) and not (img[current[0]+1][current[1]-1][0]==255):
        neighbors.append((current[0]+1,current[1]-1))

    if (current[1]>0) and (current[0]>0) and not (img[current[0]-1][current[1]-1][0]==255):
        neighbors.append((current[0]-1,current[1]-1))
    
    return neighbors
   
#AS PARENTS OF EACH NECESSARY NODE IS KNOWN, TRAVERSE BACKWARDS, DRAWING THE PATH FROM THE END TO THE STARTING NODE
def reconstruct_path(parent,start,end,copy):

        copy[end[0]][end[1]]=END_COLOR
        current=end

        while not (current==start):

            #TO CHECK IF A NON-DIAGONAL NEIGHBOR IS THE PARENT
            if parent[end[0]][end[1]]==7:
                current=(end[0],end[1]+1)

            elif parent[end[0]][end[1]]==(-7):
                current=(end[0],end[1]-1)

            elif parent[end[0]][end[1]]==5:
                current=(end[0]+1,end[1])

            elif parent[end[0]][end[1]]==(-5):
                current=(end[0]-1,end[1])

            #TO CHECK IF A DIAGONAL NEIGHBOR IS THE PARENT
            #COMMENT LINE 100-110 FOR NON_DIAG MOVEMENT DISALLOWED A* ALGORITHM
            elif parent[end[0]][end[1]]==(12):
                current=(end[0]+1,end[1]+1)

            elif parent[end[0]][end[1]]==(-12):
                current=(end[0]-1,end[1]-1)

            elif parent[end[0]][end[1]]==(2):
                current=(end[0]-1,end[1]+1)

            elif parent[end[0]][end[1]]==(-2):
                current=(end[0]+1,end[1]-1)

            copy[current[0]][current[1]]=PURPLE
            end=current
            cv.imshow("PATH",cv.resize(copy,(10*copy.shape[0],10*copy.shape[1]),interpolation = cv.INTER_AREA))
            k=cv.waitKey(1)

        end_time=time.time()

        copy[start[0]][start[1]]=START_COLOR

        #UPSCALE THE 100*100 IMAGE TO 1000*1000 IMAGE
        new=np.full((10*copy.shape[0],10*copy.shape[1],3),255,dtype=np.uint8)

        for i in range(10*copy.shape[0]):
            for j in range(10*copy.shape[1]):
                x=int(((i-(i%10))/10))
                y=int(((j-(j%10))/10))
                new[i][j][0]=copy[x][y][0]
                new[i][j][1]=copy[x][y][1]
                new[i][j][2]=copy[x][y][2]
                
        cv.imshow("PATH",new)
        k=cv.waitKey(0)

        #PRESS ESC TO EXIT
        if k == 27:                                                 
            cv.destroyAllWindows()

        return end_time

def main():

    #DEFINE A PRIORITY QUEUE AND ADD THE STARTING NODE AND IT'S F_SCORE
    open_set = PriorityQueue()
    open_set.put((0, start))

    #DEFINE 2-D ARRAYS TO STORE VALUES OF PARENTS, F_SCORE, G_SCORE OF EACH NODE INITIALISED TO INFINITY
    parent = np.full((HEIGHT,WIDTH),np.inf)

    g_score = np.full((HEIGHT,WIDTH),np.inf)
    g_score[start[0]][start[1]]=0

    f_score = np.full((HEIGHT,WIDTH),np.inf)
    f_score[start[0]][start[1]]=h(start, end)

    #DEFINE A LIST TO KEEP THE TALLY OF NEIGHBOURING NODES THAT NEED TO BE EXPLORED 
    open_set_hash = [start]

    start_time=time.time()

    #THE MAIN ALGORITHM
    while (len(open_set_hash)!=0):

        #FETCHING THE NODE WITH MINIMUM F_SCORE TO EXPLORE IT'S NEIGHBORS
        current = open_set.get()[1]
        open_set_hash.remove(current)

        #IF END IS REACHED, LOOP IS TERMINATED AND BACKTRACKING OF PATH IS DONE
        if current == end:

            copy[end[0]][end[1]]=END_COLOR

            end_time=reconstruct_path(parent, start, end, copy)
            print("TIME TAKEN = ",end_time-start_time)

            return True

        #FETCHING ALL THE POSSIBLE NEIGHBORS OF THE NODE BEING EXPLORED
        neighbors=is_defined(current,HEIGHT,WIDTH,img)


        for neighbor in neighbors:

            #IF THE NODE IS DIAGONAL/NON-DIAGONAL, TEMP_G_SCORE IS GIVEN
            if (abs(neighbor[0]-current[0])+abs(neighbor[1]-current[1]))==2:
                temp_g_score = g_score[current[0]][current[1]] + 1.414

            else:
                temp_g_score = g_score[current[0]][current[1]] + 1

            #TEMP_G_SCORE IS COMPARED WITH PREVIOUSLY STORED G_SCORE AND THE G_SCORE AND PARENTS ARE UPDATED IF REQUIRED
            if temp_g_score < g_score[neighbor[0]][neighbor[1]]:
                
                #UPDATING THE PARENT
                parent[neighbor[0]][neighbor[1]] = (current[1]-neighbor[1])*7+(current[0]-neighbor[0])*5

                #UPDATING THE G_SCORE
                g_score[neighbor[0]][neighbor[1]] = temp_g_score

                #UPDATING THE F_SCORE
                f_score[neighbor[0]][neighbor[1]] = temp_g_score + h(neighbor,end)

                #IF THE NEIGHBOR HASN'T BEEN EXPLORED YET, ADD IT TO THE OPEN SET WITH IT'S F_SCORE 
                if neighbor not in open_set_hash:

                    open_set.put((f_score[neighbor[0]][neighbor[1]], neighbor))
                    open_set_hash.append(neighbor)

                    #NODES WHICH HAVE NOT BEEN EXPLORED YET BUT PARENTS AND F_SCORE UPDATED
                    copy[neighbor[0]][neighbor[1]]=GREEN

        #NODES WHICH HAVE BEEN EXPLORED
        if current != start:
            copy[current[0]][current[1]]=RED
        cv.imshow("PATH",cv.resize(copy,(10*copy.shape[0],10*copy.shape[1]),interpolation = cv.INTER_AREA))
        k=cv.waitKey(1)
    return False

main()
