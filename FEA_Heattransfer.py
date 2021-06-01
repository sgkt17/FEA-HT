class Finite_Eliment_Solver:

      # rectangular system x[0,1] y[0,1] divided 4*2 element
      # 4 node for each element, 1 degree of freedom
      # Gausss regendre intergration
      # Essential B.C u(1,y)=u(x,1) = 0
      # Natural B.C au/ax(0,y) = au/ay(1,y) = 0
      
        def __init__(self):

            NELEM, NNPE, NNODE = 4*2, 4, 1  
                 
            self.Mesh(1, 1, 15, 15) ## Mesh input (lenx, leny, number of x element, number of y element)
 
        def Mesh(self, xdim, ydim, xelem, yelem):

            temp_y = 0 
            globalcoordinate = [] ## globalcoordinate[i-1] ===> i element's global coordinate
            
            for i in range(yelem +1):

                temp_x = 0

                for j in range(xelem + 1):
                        
                    temp = (temp_x, temp_y)
                    globalcoordinate.append(temp)

                    temp_x = temp_x + xdim / int(xelem)
                    
                temp_y = temp_y + (ydim / int(yelem))
            
            globalstiffness = [[0]*((xelem+1)*(yelem+1)) for i in range(((xelem+1)*(yelem+1)))]
            globalloadf = [0 for i in range((xelem+1)*(yelem+1))]
            globalloadq = [0 for i in range((xelem+1)*(yelem+1))]
            
            for element in range(1, xelem*yelem + 1):
                
                for i in range(1,5):
                    
                    globalpointf = self.Connectivity_matrix(element, i, xelem)        
                    globalloadf[globalpointf-1] += self.Fstiff(element, globalcoordinate, xelem,i)
                    
                    for j in range(1,5):
               
                        globalpoint_row = self.Connectivity_matrix(element, i, xelem)
                        globalpoint_column = self.Connectivity_matrix(element, j, xelem)
                        
                        globalstiffness[globalpoint_row-1][globalpoint_column-1] += self.Estiff(element,globalcoordinate,xelem, i, j)

            
            globalstiffness, globalloadf = self.Boundary_condition(globalstiffness, globalloadf,globalcoordinate,xelem,yelem)

            for i in range((xelem+1)*(yelem+1)):

                    globalstiffness[i].append(globalloadf[i])
        
            answer_array = self.Gaussian_Elimination(globalstiffness)

            self.graff(answer_array, globalcoordinate,xelem,yelem)
            
            return None
                                                             
        def Find_global_location(self, globalcoordinate, globalpoint):

            globalpoint_location = globalcoordinate[globalpoint-1]
            
            return globalpoint_location
                                                             
        def Connectivity_matrix(self, element, localpoint, xelem):

            element = element - 1 
            localpoint = localpoint - 1
            
            if localpoint in [0,1]:
                    
                    globalpoint = element + localpoint + int(element / xelem) 
                    
            elif localpoint == 2:

                    globalpoint = element + localpoint + int(element / xelem) + xelem
                    
            elif localpoint == 3:

                    globalpoint = element + localpoint + int(element / xelem) + xelem - 2

            else:
                    print("Select local number => 1,2,3,4 ")
            
            return globalpoint+1
        
        def Estiff(self,element,globalcoordinate,xelem, i, j):
            
            intergration_point = [-0.57735022692, 0.57735022692]

            s11 = [[0,0,0,0] for i in range(4)]
            s22 = [[0,0,0,0] for i in range(4)]
            
            for k in range(4):
                
                for l in range(4):
                    
                    for a in intergration_point:
                        
                        for b in intergration_point:

                            res = self.Jacobbian(element,globalcoordinate,xelem,a,b)

                            detj = np.linalg.det(res)

                            res = np.linalg.inv(np.array(res))
                            
                            partialzeta =  [-1/4*(1-b), 1/4*(1-b), 1/4*(1+b),-1/4*(1+b)] ## shapefunction partial zeta
                            partialeta =  [-1/4*(1-a),-1/4*(1+a), 1/4*(1+a), 1/4*(1-a)]  ## shapefunction partial eta

                            s11[k][l] += (res[0][0]*partialzeta[k] + res[0][1]*partialeta[k] )*(res[0][0]*partialzeta[l]+ res[0][1]*partialeta[l])*detj
                            s22[k][l] += (res[1][0]*partialzeta[k] + res[1][1]*partialeta[k] )*(res[1][0]*partialzeta[l]+ res[1][1]*partialeta[l])*detj

            estiff = np.array(s11) + np.array(s22)
            
            return estiff[i-1][j-1]
        
        def Fstiff(self,element, globalcoordinate, xelem,i):
                     
            intergration_point = [-0.57735022692, 0.57735022692]
            
            Fstiff = [0,0,0,0]
            
            for k in range(4):
                for a in intergration_point:
                    for b in intergration_point:
                
                        res = self.Jacobbian(element,globalcoordinate,xelem,a,b)
                        detj = np.linalg.det(res)
                        shaeffunc = [1/4*(1-a)*(1-b), 1/4*(1+a)*(1+b), 1/4*(1+a)*(1-b), 1/4*(1-a)*(1+b)] 
                        
                        Fstiff[k] += shaeffunc[k]*detj
            
            return Fstiff[i-1]
                
        def Jacobbian(self, element, globalcoordinate, xelem, a,b):

            x1,y1 = globalcoordinate[self.Connectivity_matrix(element, 1, xelem)-1]
            x2,y2 = globalcoordinate[self.Connectivity_matrix(element, 2, xelem)-1]
            x3,y3 = globalcoordinate[self.Connectivity_matrix(element, 3, xelem)-1]
            x4,y4 = globalcoordinate[self.Connectivity_matrix(element, 4, xelem)-1]
            
            res = [[1/4*(-x1*(1-b) + x2*(1-b) + x3*(1+b) - x4*(1+b)),
                    1/4*(-y1*(1-b) + y2*(1-b) + y3*(1+b) - y4*(1+b))],
                   [1/4*(-x1*(1-a) - x2*(1+a) + x3*(1+a) + x4*(1-a)),
                    1/4*(-y1*(1-a) - y2*(1+a) + y3*(1+a) + y4*(1-a))]]

            return res



        def Boundary_condition(self,globalstiffness, globalloadf,globalcoordinate,xelem,yelem):

            for i in range((xelem+1)*(yelem+1)):
                
                tempx, tempy = globalcoordinate[i]

                
                if tempy > 0.99999 or tempx > 0.99999:

                    
                    U = 0 ## Difined essential boundary condition 

                    for j in range((xelem+1)*(yelem+1)):

                        if j == i :

                            globalstiffness[i][j] = 1
                            globalloadf[j] = U
                            
                        else :
                        
                            globalloadf[j] = globalloadf[j] - U*globalstiffness[i][j]
                            globalstiffness[i][j] = 0

                    for k in range((xelem+1)*(yelem+1)):

                        if k == i :

                            continue

                        else:

                            globalstiffness[k][i] = 0
               
            return globalstiffness, globalloadf
        
        def Gaussian_Elimination(self, stiffness_matrix):

            n_dimenstion = len(stiffness_matrix)

            for i in range(0, n_dimenstion):
                    
                maximum_element = abs(stiffness_matrix[i][i])
                maximum_row = i
                
                for k in range(i+1, n_dimenstion):
                        
                    if abs(stiffness_matrix[k][i]) > maximum_element:
                            
                        maximum_Element = abs(stiffness_matrix[k][i])
                        maximum_Row = k

                for k in range(i, n_dimenstion+1):
                        
                    temp = stiffness_matrix[maximum_row][k]
                    stiffness_matrix[maximum_row][k] = stiffness_matrix[i][k]
                    stiffness_matrix[i][k] = temp

  
                for k in range(i+1, n_dimenstion):
                        
                    constant = -stiffness_matrix[k][i]/stiffness_matrix[i][i]
                    
                    for j in range(i, n_dimenstion+1):
                            
                        if i == j:
                                
                            stiffness_matrix[k][j] = 0
                            
                        else:
                                
                            stiffness_matrix[k][j] += constant * stiffness_matrix[i][j]

            answer_array = [0 for i in range(n_dimenstion)]
            
            for i in range(n_dimenstion-1, -1, -1):
                    
                answer_array[i] = stiffness_matrix[i][n_dimenstion]/stiffness_matrix[i][i]
                
                for k in range(i-1, -1, -1):
                        
                    stiffness_matrix[k][n_dimenstion] -= stiffness_matrix[k][i] * answer_array[i]

            return answer_array
        
        def graff(self, answer_array, globalcoordinate,xelem,yelem):

            sct = []
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            
            for i in range((xelem+1)*(yelem+1)):

                    tempx, tempy = globalcoordinate[i]
                    ax.scatter(tempx,tempy,answer_array[i],color = 'black')

            plt.show()
                
        
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    FEA = Finite_Eliment_Solver()
