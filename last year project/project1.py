import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Particles:
    def __init__(self,n_clusters,X):
        xshape0=X.shape[0]
        
        self.clusters_arr=random.sample(set(X),n_clusters)
        self.clusters_arr=sorted(self.clusters_arr)
        #print(self.clusters_arr)
        self.pixels_arr=np.zeros(xshape0)
        self.pbest_value = float('inf')
        self.pbest_position=self.clusters_arr
        self.p_value=float('inf')
        self.velocity = np.zeros(n_clusters)
        

    
    def move(self):
        #print("position is",self.clusters_arr)
        v=np.array(self.clusters_arr)+np.array(self.velocity)
        v=v.astype(int)
        for i in range(len(v)):
            if(v[i]<0):
                v[i]=0
            elif(v[i]>255):
                v[i]=255       
        self.clusters_arr = sorted(list(v))

class Pso:
    
    def __init__(self,n_clusters,X,a1=1,a2=.17,a3=6,w=.5,c1=2,c2=2,particle=10,iteration=20):
        self.W=w
        self.c1=c1
        self.c2=c2
        self.a1=a1
        self.a2=a2
        self.a3=a3
        
        self.X=X
        self.n_particles=particle
        self.n_iter=iteration
        self.gbest_value=float('inf')
        self.n_clusters=n_clusters
        self.gbest_position=np.zeros(self.n_clusters)
        self.n_pixels=X.shape[0]
        self.particles=[Particles(n_clusters,X) for i in range(self.n_particles)]
        self.final_arr=[]
        

    def pso_cluster(self):
        for t in range(self.n_iter):
            for particle in self.particles:#particles is the array of all particle
                for pixel in range(self.n_pixels):
                    cluster=self.calculate_cluster(self.X[pixel],particle)
                    particle.pixels_arr[pixel]=cluster  
                fit_val=self.fitness(particle)
                particle.p_val=fit_val
                self.set_pbest(particle,fit_val)
            self.set_gbest()
            self.move_particles()
            print((t+1)*100/self.n_iter,"%")
  
    #this function calculate a pixel is belongs to which cluster for a perticle    
    def calculate_cluster(self,pixel,particle):
        min=-1
        value=0
        for index in range(self.n_clusters):
            dist=abs(pixel-particle.clusters_arr[index])
            if(min!=-1):
                if(dist<min):
                    min=dist
                    value=index
            else:
                min=dist
                value=index   
        return value
      
    
    
    #mean error            
    def me(self,particle):
        sum_f=0
        p=particle.pixels_arr.shape[0]
        for pixel in range(p):
            sum_f+=abs(int(self.X[int(pixel)])-int(particle.clusters_arr[int(particle.pixels_arr[int(pixel)])]))
        return(sum_f/p)
    
            
    #function to calculate the minimum distance between clusters
    def calculate_inter_cluster_separation(self,particle):
        arr=[]
        for i in range(self.n_clusters-1):
            for j in range(i+1,self.n_clusters):
                dist=abs(int(particle.clusters_arr[i])-int(particle.clusters_arr[j]))
                if(dist==0):
                    return -256
                arr.append(dist)
        return(min(arr))  
   
            

    #function to calculate the maximum distance of any pixel within any clusters    
    def calculate_intra_cluster_distance(self,particle): 
        clusters=[[]]*self.n_clusters
        arr=[]
        for pixel in range(particle.pixels_arr.shape[0]):
            clusters[int(particle.pixels_arr[pixel])].append(self.X[pixel])
        for cluster in clusters:
            centeroid=particle.clusters_arr[clusters.index(cluster)]
            l=len(cluster)
            s=0
            for pixel in cluster:
                s+=abs(int(pixel)-int(centeroid))
            s/=l
            arr.append(s)
        return(max(arr))

    #if a cluster is empty then no meaning to use this clusters. is_empty_cluster check this    
    def is_empty_cluster(self,particle):
        for i in range(self.n_clusters):
            if(list(particle.pixels_arr).count(i)==0):
                return 1
        return 0    
            
        
        
    def fitness(self,particle):
        
        inter_cluster_separation=self.calculate_inter_cluster_separation(particle)
        intra_cluster_distance=self.calculate_intra_cluster_distance(particle) 
        ME=self.me(particle)
        #print(MSE,inter_cluster_separation,intra_cluster_distance)
        is_empty=self.is_empty_cluster(particle)
        fit_val=self.a1*intra_cluster_distance+self.a2*(255-int(inter_cluster_separation))+self.a3*ME+(1000000000)*is_empty
        #print(fit_val)
        return(fit_val)
        
   
    
    
    
    def move_particles(self):

        for particle in self.particles:
            
           
            new_velocity = (self.W*np.array(particle.velocity)) + (self.c1*random.random()) * (np.array(particle.pbest_position) + np.array(particle.clusters_arr)*(-1)) + \
                            (random.random()*self.c2) * (np.array(self.gbest_position) + np.array(particle.clusters_arr)*(-1))
            particle.velocity = list(new_velocity)
            #print("velocity is ",particle.velocity)
            
    
            particle.move()
            
            
            
    def set_pbest(self,particle,fit_val):
        if(particle.pbest_value > float(fit_val)):
            particle.pbest_value = float(fit_val)
            particle.pbest_position=particle.clusters_arr
                
            

    def set_gbest(self):
        for particle in self.particles:
            if(self.gbest_value > particle.pbest_value):
                self.gbest_value =  particle.pbest_value
                self.gbest_position = particle.pbest_position
                self.final_arr=particle.pixels_arr
        

#convert RGB image to grayscle image                 
def makeGray(img):
    X=img[ :,: , 0]
    Y=img[ :,: , 1]
    Z=img[ :,: , 2]
    new_img=0.2989*X+0.5870*Y+0.1140*Z
    return new_img

def create_image(K):
    path1="/home/roshan/Documents/LAST YEAR project/brain_image_seg/useful data(image)/t1_icbm_normal_1mm_pn3_rf20.rawb"
    f=open(path1,"rb")
    f1=f.read()
    img=[]
    k=K
    for i in range(181):
        x=[]
        for j in range(217):
            x.append(f1[39277*k+181*i+j])
        img.append(x)
    img=np.array(img)    
    img=img.astype(np.uint8)
    sx,sy = img.shape
    plt.imshow(img,cmap="gray")
    img=img.reshape(sx*sy).astype(int)
    for i in range(sx*sy):
        if(img[i]>=165):
            img[i]=0
    return (img,sx,sy) 

def show_image(img1,img2,sx,sy,str1="our segmented image(4 clusters)",str2=" original tissue pattern(10 clusters)"):
    fig, ax = plt.subplots(1, 2, figsize = (12, 8))
    ax[0].imshow(img1.reshape(sx,sy),cmap="gray")
    ax[0].set_title(str1)
    ax[1].imshow(img2.reshape(sx,sy),cmap="gray")
    ax[1].set_title(str2)
    for ax in fig.axes:
        ax.axis('off')
    plt.tight_layout()

def accuracy_4(x,z,sx,sy):  
    path2="/home/roshan/Documents/LAST YEAR project/brain_image_seg/useful data(image)/phantom_1.0mm_normal_crisp.rawb"
    f2=open(path2,"rb")
    f2=f2.read()
    img=[]
    k=x
    z=z
    for i in range(181):
        x=[]
        for j in range(217):
            x.append(f2[39277*k+181*i+j])
        img.append(x)
    img=np.array(img)
    img1=img.reshape(sx*sy) 
    test=np.ndarray.tolist(img1)
    no_of_point=0
    true_point=0
    for i in range(sx*sy):
        if(test[i]==0):
            if(z[i]==0):
                true_point+=1
            no_of_point+=1
        elif(test[i]==1):
            if(z[i]==1):
                true_point+=1
            no_of_point+=1
        elif(test[i]==2):
            if(z[i]==2):
                true_point+=1
            no_of_point+=1
        elif(test[i]==3):
            if(z[i]==3):
                true_point+=1
            no_of_point+=1
    accuracy=((true_point*100)/no_of_point)
    
    print("accuracy is {}%".format(accuracy)) 
    show_image(z,img1,sx,sy)

def remove_bg(img_bg,it,sx,sy):
    new_img=img_bg.copy()
    factor=1
    for j in range(it):
        if(j>0):
            img_bg=new_img.copy()
        for i in range(sx*sy):
            if(img_bg[i]<=40):
                total=0
                count=0
                try:
                    total+=img_bg[i-1]
                    count+=1
                except:
                    pass
                try:
                    total+=img_bg[i+1]
                    count+=1
                except:
                    pass
                try:
                    total+=img_bg[i-sy-1]
                    count+=1
                except:
                    pass
                try:
                    total+=img_bg[i-sy+1]
                    count+=1
                except:
                    pass
                try:
                    total+=img_bg[i-sy]
                    count+=1
                except:
                    pass
                try:
                    total+=img_bg[i+sy-1]
                    count+=1
                except:
                    pass
                try:
                    total+=img_bg[i+sy+1]
                    count+=1
                except:
                    pass
                try:
                    total+=img_bg[i+sy]
                    count+=1
                except:
                    pass
                avarage=total//count
                if(avarage>40):
                    new_img[i]=int(img_bg[i]+factor*(avarage-img_bg[i]))
    return new_img 

def main():
    img_num=int(input("enter the image number(1-180):"))
    if(img_num>180 or img_num<0):
        print("invalid input !")
        print("try again")
    else:
        print("please wait while computation ...")
        (img,sx,sy)=create_image(img_num)
        img=remove_bg(img,10,sx,sy)
        pso=Pso(4,img,a1=.9,a2=.17,a3=6,w=.5,c1=2,c2=2,particle=10,iteration=20)
        pso.pso_cluster()
        final_image=pso.final_arr
        show_clusters(final_image,sx,sy)
        accuracy_4(img_num,final_image,sx,sy)

def show_clusters(final_image,sx,sy):
    fig,ax=plt.subplots(2,2,figsize=(12,12))
    new_img=final_image.copy()
    for j in range(sx*sy):
        if(new_img[j]==3):
            new_img[j]=0
        else:
            new_img[j]=255
    ax[0][0].imshow(new_img.reshape(sx,sy),cmap="gray")
    ax[0][0].set_title("white matter only")
    new_img=final_image.copy()
    for j in range(sx*sy):
        if(new_img[j]==2):
            new_img[j]=0
        else:
            new_img[j]=255
    ax[0][1].imshow(new_img.reshape(sx,sy),cmap="gray") 
    ax[0][1].set_title("gray matter only")
    new_img=final_image.copy()
    for j in range(sx*sy):
        if(new_img[j]==1):
            new_img[j]=0
        else:
            new_img[j]=255
    ax[1][0].imshow(new_img.reshape(sx,sy),cmap="gray") 
    ax[1][0].set_title("CSF only")
    new_img=final_image.copy()
    for j in range(sx*sy):
        if(new_img[j]==0):
            new_img[j]=0
        else:
            new_img[j]=255
    ax[1][1].imshow(new_img.reshape(sx,sy),cmap="gray") 
    ax[1][1].set_title("background")

main()




