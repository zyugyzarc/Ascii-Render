# distutils: language = c++

import os, sys
from libc.stdlib cimport calloc, free #, system
from libc.stdio cimport printf
from libc.math cimport sin, cos, sqrt
from libcpp cimport bool
from time import perf_counter as time, sleep

if "DISPLAY" in os.environ: import pyautogui as pg # to get mouse position

cdef int FPS = 12;
cdef int COLOR = 1; # BOOL : ENUM(0, 1)
cdef int ASPECT = 1;

if os.environ.get("FPS"):     FPS = <int>int(os.environ["FPS"])
if os.environ.get("NOCOLOR"): COLOR = 0
if os.environ.get("ASPECT"): ASPECT = <int>int(os.environ["ASPECT"])

cdef struct vec:
    float x
    float y
    float z

cdef struct ivec:
    int x
    int y
    int z

cdef int w = <int>((os.get_terminal_size()[0])/ASPECT)
cdef int h = <int>(os.get_terminal_size()[1] - 1)
cdef int TEXSIZE = 160

cdef ivec* tex_buffer = <ivec*> calloc( TEXSIZE*TEXSIZE *3, sizeof(ivec) )

printf("\nsize: %d x %d\n", w, h)

cdef char* screen = <char*> calloc( w*h, sizeof(char) );
cdef ivec* color = <ivec*> calloc( w*h, sizeof(ivec) );

cdef void norm( vec& v , float s = 1):
    cdef float m = sqrt( v.x * v.x + v.y * v.y + v.z * v.z )
    v.x = v.x /m *s
    v.y = v.y /m *s
    v.z = v.z /m *s

cdef float mag( vec v ):
    return sqrt( v.x * v.x + v.y * v.y + v.z * v.z )

cdef float dist( vec a, vec b):
    cdef vec v = vec(a.x-b.x, a.y-b.y, a.z-b.z )
    return sqrt( v.x * v.x + v.y * v.y + v.z * v.z )

cdef ivec to_ivec( vec v ):
    return ivec(<int>v.x, <int>v.y, <int>v.z)

cdef vec to_vec( ivec v ):
    return vec(v.x, v.y, v.z)

cdef void clear():

    cdef int i;
    for i from 0 <= i < w*h by 1:
        screen[i] = b" "
        color[i] = ivec(255, 255, 255)
    printf("\x1b[H\x1b[2J")


cdef void show():
    cdef int i, j;
    for i from 0 <= i < w*h by 1:
        if COLOR==1 and ( color[i].x != 255 or color[i].y != 255 or color[i].z != 255 ):
            printf("\x1b[38;2;%d;%d;%dm",
                color[i].x,
                color[i].y,
                color[i].z
            )

            for j from 0 <= j < ASPECT:
                printf("%c", screen[i])
            printf("\x1b[0m")

        else:
            for j from 0 <= j < ASPECT:
                printf("%c", screen[i])

        if i%w == 0:
            printf("\n")

    printf('\n')

cdef void point(ivec p, char c, ivec col):
    if col.x >= 0:
        if 0 < p.y and 0 < p.x and p.y < h and p.x < w:
            screen[p.y*w + p.x] = c
            color[ p.y*w + p.x] = col

#@cython.cdivision(True)
cdef void line(ivec p1, ivec p2, char ch, ivec col):
    
    cdef float m, c
    cdef int i =0

    if p1.x != p2.x:

        m = (p2.y-p1.y)/(p2.x-p1.x)
        c = p1.y - m * p1.x
        
        if p1.x < p2.x:
            for i from p1.x < i < p2.x by 1:
                point( ivec(i, <int>(i*m+c), 0), ch, col )
                

        if p2.x < p1.x:
            for i from p2.x < i < p1.x by 1:
                point( ivec(i, <int>(i*m+c), 0), ch, col )

    if p1.y != p2.y:

        m = (p2.x-p1.x)/(p2.y-p1.y)
        c = p1.x- m*p1.y
        
        if p1.y < p2.y:
            for i from p1.y < i < p2.y by 1:
                point( ivec( <int>(i*m+c), i, 0), ch, col)

        if p2.y < p1.y:
            for i from p2.y < i < p1.y by 1:
                point( ivec( <int>(i*m+c), i, 0), ch, col)

#@cython.cdivision(True)
cdef vec interpolate_tex(ivec p, ivec p1, ivec p2, ivec p3):

    cdef vec q
    if  (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y) == 0:
        return vec(0, 0, 0)

    q.x = ( (p2.y - p3.y) * (p.x - p3.x) + (p3.x - p2.x) * (p.y - p3.y) )/( (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y) )
    q.y = ( (p3.y - p1.y) * (p.x - p3.x) + (p1.x - p3.x) * (p.y - p3.y) )/( (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y) )
    q.z = 1 - q.x - q.y

    q.x = ((q.x if q.x < 1 else 0.99) if q.x > 0 else 0.01)
    q.y = ((q.y if q.y < 1 else 0.99) if q.y > 0 else 0.01)
    q.z = ((q.z if q.z < 1 else 0.99) if q.z > 0 else 0.01)
    #printf("q<%d %d %d>", q.x, q.y ,q.z)
    return q


cdef ivec interpolate(vec p, vec t1, vec t2, vec t3, int tex_id, float z1, float z2, float z3):

    cdef ivec p_, t_
    cdef vec m, t, d1, d2, d3
    cdef int i
    cdef int s = 2

    #if tex_id == 2:  # Raw tex-interpolated channel
    # 
    #    p_ = ivec( <int>(255*p.x), <int>(255*p.y), <int>(255*p.z) )
    #    return p_
    #    
    # 
    if True:

        t.x  = t1.x * p.x + t2.x * p.y + t3.x * p.z
        t.y  = t1.y * p.x + t2.y * p.y + t3.y * p.z

        t.x = ((t.x if t.x < 1 else 1) if t.x > 0 else 0)
        t.y = ((t.y if t.y < 1 else 1) if t.y > 0 else 0)
        
        #printf("(%f %f)(%f %f)(%f %f) -> [%d %d]\n", t1.x, t1.y, t2.x, t2.y, t3.x, t3.y, t.x*255, t.y*255)
        #return ivec(<int>(t.x * 255), <int>(t.y * 255), 255)

        t_.x = <int> ((t.x * (TEXSIZE)))
        t_.y = <int> ((t.y * (TEXSIZE)))

        
        i = <int>(( t_.y * (TEXSIZE) + t_.x ))
        #printf("[%d](%f %f)(%d %d)\n", i, t.x, t.y, t_.x, t_.y)
        

        if 0 <= i < TEXSIZE*TEXSIZE:
            return tex_buffer[ (tex_id-1) * TEXSIZE * TEXSIZE +  i ]
        else:
            return ivec(-1, 0, 0) # dont render the area
            #return tex_buffer[ (tex_id) * TEXSIZE * TEXSIZE - 1 ]
            #return ivec(255, 0, 255) # for debugging

    

#@cython.cdivision(True)
cdef void triangle(ivec p1, ivec p2, ivec p3, vec t1, vec t2, vec t3, float lum, ivec col, float z1, float z2, float z3):

    cdef int[3] xi = [p1.x, p2.x, p3.x]
    cdef int[3] yi = [p1.y, p2.y, p3.y]
    cdef vec[3] ti = [t1, t2, t3]
    cdef ivec p4
    cdef float m, c
    cdef int i, j = 0
    cdef int tex = 0
    cdef char* chars = ".:`\'-,;~_!\"?c\\^<>|=sr1Jo*(C)utia3zLvey75jST{lx}IfY]qp9n0G62Vk8UXhZ4bgdPEKA$wQm&#HDR@WNBM"
    cdef int clen = 88//2
    cdef char ch = chars[<int>lum]

    cdef float col_sc = 0.5
    
    cdef float l = 1
    lum *= 2

    if col.x < 0:
        tex = -col.x

    for j from 1<= j < 3 by 1:
        for i from 1<= i < 3 by 1:
            if yi[i-1] > yi[i]:
                yi[i-1], yi[i] = yi[i], yi[i-1]
                xi[i-1], xi[i] = xi[i], xi[i-1]
                ti[i-1], ti[i] = ti[i], ti[i-1]

    p1.x, p2.x, p3.x = xi[0], xi[1], xi[2]
    p1.y, p2.y, p3.y = yi[0], yi[1], yi[2]
    t1, t2, t3 = ti[0], ti[1], ti[2]
    
    cdef float m1, c1, m2, c2
    
    if p1.y == p3.y:
        line(p1, p2, ch, col)
        line(p1, p3, ch, col)
    else:
        m = (p1.x - p3.x)/(p1.y - p3.y)
        c = p3.x - m * p3.y

        p4.x = <int>(m * p2.y + c)
        p4.y = p2.y

        if p1.y == p2.y:
            line(p1, p2, ch, col)
        else:
            # draw flat bottom triangle : p1 | p2 -- p4
            m1 = (p1.x - p2.x)/(p1.y - p2.y)
            c1 = p1.x - m1*p1.y

            m2 = (p1.x - p4.x)/(p1.y - p4.y)
            c2 = p1.x - m2*p1.y

            for i from p1.y <= i <= p2.y by 1:
                for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                    l = 1
                    if tex:
                        col = interpolate(
                                interpolate_tex( ivec(j, i, 0), p1, p2, p3), 
                                t1, t2, t3, tex,
                                z1, z2, z3
                            )
                        l = color_char(col, col_sc)/255
                    l = (lum * l)
                    l = l * clen
                    point( ivec(j, i, 0), chars[<int>l], col)

                for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                    l = 1
                    if tex:
                        col = interpolate(
                                interpolate_tex( ivec(j, i, 0), p1, p2, p3),
                                t1, t2, t3, tex,
                                z1, z2, z3
                            )
                        l = color_char(col, col_sc)/255
                    l = (lum * l)
                    l = l * clen
                    point( ivec(j, i, 0), chars[<int>l], col)

        if p2.y == p3.y:
            line(p2, p3, ch, col)
        else:
            # draw flat top tiangle : p2 -- p4 | p3
            m1 = (p3.x - p2.x)/(p3.y - p2.y)
            c1 = p3.x - m1*p3.y

            m2 = (p3.x - p4.x)/(p3.y - p4.y)
            c2 = p3.x - m2*p3.y

            for i from p2.y <= i <= p3.y by 1:
                for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                    l = 1
                    if tex:
                        col = interpolate(
                                interpolate_tex( ivec(j, i, 0), p1, p2, p3),
                                t1, t2, t3, tex,
                                z1, z2, z3
                            )
                        l = color_char(col, col_sc)/255
                    l = (lum * l)
                    l = l * clen
                    point( ivec(j, i, 0), chars[<int>l], col)

                for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                    l = 1
                    if tex:
                        col = interpolate(
                                interpolate_tex( ivec(j, i, 0), p1, p2, p3),
                                t1, t2, t3, tex,
                                z1, z2, z3
                            )
                        l = color_char(col, col_sc)/255
                    l = (lum * l)
                    l = l * clen
                    point( ivec(j, i, 0), chars[<int>l], col)

#cdef void quad(ivec p1, ivec p2, ivec p3, ivec p4, char c, ivec col):
#    triangle(p1, p2, p3, c, col)
#    triangle(p2, p3, p4, c, col)


cdef float deg(float x): return 180 * x / 3.14159
cdef float rad(float x): return 3.14159 * x / 180


cdef void mat_mul(float* m1, int a1, int b1, float* m2, int a2, int b2, float* m3):
    
    cdef int i, j, k, l
    cdef float s

    # m1 is a1 x b1 matrix
    # m2 is a2 x b2 matrix

    if b1 != a2: printf("invalid matrix %dx%d x %dx%d\n", a1, b1, a2, b2)

    for i from 0 <= i < a1 by 1:
        for j from 0 <= j < b2 by 1:
            s=0
            for k from 0 <= k < a2 by 1:
                s = s + m1[i*b1+k] * m2[j+k*b2]
            m3[i*b2+j] = s

cdef void rotate_point(vec& p_, vec r, bool reverse = False):
    
    cdef float p[3]
    p[0], p[1], p[2] = p_.x, p_.y, p_.z

    cdef float Rx[9]
    cdef float Ry[9]
    cdef float Rz[9]
    cdef int i

    Rx[:] = [1,       0,       0, 
             0,  cos(r.x), -sin(r.x),
             0,  sin(r.x),  cos(r.x)]

    Ry[:] = [cos(r.y),  0, -sin(r.y), 
             0     ,  1,       0,
             sin(r.y),  0,  cos(r.y)]

    Rz[:] = [cos(r.z), -sin(r.z),  0, 
             sin(r.z),  cos(r.z),  0,
             0     ,       0,  1]

    cdef float P1[3]
    cdef float P2[3]
    
    if not reverse:
        mat_mul(Rx, 3, 3, p , 3, 1, P1)
        mat_mul(Ry, 3, 3, P1, 3, 1, P2)
        mat_mul(Rz, 3, 3, P2, 3, 1, p )
    else:
        mat_mul(Rz, 3, 3, p , 3, 1, P1)
        mat_mul(Ry, 3, 3, P1, 3, 1, P2)
        mat_mul(Rx, 3, 3, P2, 3, 1, p )


    p_.x, p_.y, p_.z = p[0], p[1], p[2]

cdef void move_point(vec& p, vec q):
    p.x += q.x
    p.y += q.y
    p.z += q.z

cdef vec project_point(vec& p, float sx, float sy):

    cdef float distance = 5.0
    cdef float z = 1/p.z
    p.x = p.x * z * sx + w/2
    p.y = p.y * z * sy + h/2
    p.z = z



cdef int color_char(ivec& c, float scale = 1):
    
    if c.x < 0:
        return 50

    cdef int m
    m = ( c.x if c.x > c.y else c.y )
    m = ( m   if m   > c.z else c.z )
    
    c.x = <int>( ( 200/m * c.x * scale) + (m * 50/255 * c.x/255 * scale) + c.x * (1-scale) )
    c.y = <int>( ( 200/m * c.y * scale) + (m * 50/255 * c.y/255 * scale) + c.y * (1-scale) )
    c.z = <int>( ( 200/m * c.z * scale) + (m * 50/255 * c.z/255 * scale) + c.z * (1-scale) )

    return m


#clear()

cdef vec cam_pos = vec(0, 0, 0)
cdef vec cam_rot = vec(0, 0, 0)
cdef vec light = vec(5, 5, 0)

#cpdef class Mesh:
cdef class Mesh:

    cdef vec* points
    cdef vec* normals;
    cdef vec* texels;
    cdef ivec* faces;
    cdef bool* ftex;
    cdef int* fmat;
    cdef int* nmap;
    cdef ivec* tmap;

    cdef vec* vbuf;
    cdef float* dbuf;
    cdef int* dbuf_idx;

    #for large models
    #cdef vec points[2048];
    #cdef vec normals[4096];
    #cdef vec texels[4096];
    #cdef ivec faces[4096];
    #cdef int fmat[4096];
    #cdef int nmap[4096];
    #cdef ivec tmap[4069]

    #cdef vec vbuf[2048];
    #cdef float dbuf[4096];
    #cdef int dbuf_idx[4096];
    

    cdef ivec pallete[16];
    cdef int vcount
    cdef int fcount

    cdef int i
    
    cdef public vec pos;
    cdef public vec rot;
    

    cdef vec apply_transform(self, vec& p):

        rotate_point(p, self.rot)
        move_point(p, self.pos)
        return self.world_transform(p)

    cdef vec world_transform(self, vec& p):

        cdef vec temp
        temp.x, temp.y, temp.z = -cam_pos.x, -cam_pos.y, -cam_pos.z
        move_point(p, temp)

        temp.x, temp.y, temp.z = -cam_rot.x, -cam_rot.y, -cam_rot.z
        rotate_point(p, temp, reverse=True)
        
        return p

    cdef char chars[37]

    cdef void project_points(self):
        
        cdef int j
        cdef vec p
        cdef int l = ( w if w < h else h ) * 4
        for j from 0<= j < self.vcount by 1:

            p = self.points[j]
            self.apply_transform(p)
            project_point(p, l/2, l/4)
            self.vbuf[j] = p

    cdef void order_depth(self):

        cdef ivec face
        cdef int j, k
        for j from 0<= j < self.fcount by 1: # fill depth buf
                
                face = self.faces[j]
                self.dbuf[j] = (self.vbuf[ face.x ].z + 
                                self.vbuf[ face.y ].z +
                                self.vbuf[ face.z ].z )/3
                self.dbuf_idx[j] = j

        
        for j from 0<= j < self.fcount by 1: # sort depth buf
            for k from 1<= k < self.fcount by 1:
                if self.dbuf[k-1] > self.dbuf[k]:
                    self.dbuf[k-1], self.dbuf[k] = self.dbuf[k], self.dbuf[k-1]
                    self.dbuf_idx[k-1], self.dbuf_idx[k] = self.dbuf_idx[k], self.dbuf_idx[k-1]

    cpdef void render(self):
        
        self.project_points()
        self.order_depth()

        cdef int j, k
        cdef vec n, l
        cdef float lum
        cdef ivec col
        cdef char c
        
        for j from 0 <= j < self.fcount by 1:
            
            k = self.dbuf_idx[j]

            if (    (self.vbuf[ self.faces[k].x ].x < 0 or self.vbuf[ self.faces[k].x ].x > w)\
                 or (self.vbuf[ self.faces[k].x ].y < 0 or self.vbuf[ self.faces[k].x ].y > h) )\
               and( (self.vbuf[ self.faces[k].y ].x < 0 or self.vbuf[ self.faces[k].y ].x > w)\
                 or (self.vbuf[ self.faces[k].y ].y < 0 or self.vbuf[ self.faces[k].y ].y > h) )\
               and( (self.vbuf[ self.faces[k].z ].x < 0 or self.vbuf[ self.faces[k].z ].x > w)\
                 or (self.vbuf[ self.faces[k].z ].y < 0 or self.vbuf[ self.faces[k].z ].y > h) ):
                   #printf("skipped %d\n", j)
                   continue # occlusion cull

            n = self.normals[ self.nmap[k] ]
            l = light
            
            self.apply_transform(n)
            self.world_transform(l)
            norm(l, 1)
            # get color from materials (mtl file) 
            
            # color transform:
            # color in (RGB) -> (HSV)
            # V channel => lum, color (HS V=100%) -> (RGB)
            col = self.pallete[ self.fmat[k] ]

            if self.ftex[ self.fmat[k] ]:
                col = ivec( -self.fmat[k], 0, 0)
            
            lum = ( n.x * l.x + n.y * l.y + n.z * l.z )
            lum = lum * color_char(col)/255 
            lum = lum if lum > 0 else 0

            #lum = lum if lum < 36 else 36
            #c = self.chars[<int>lum]

            # Draw face
            
            #printf("(%d %d %d)\n", self.tmap[k].x, self.tmap[k].y, self.tmap[k].z,)


            triangle(
                ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0),
                ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0),
                ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0),
                self.texels[self.tmap[k].x], self.texels[self.tmap[k].y], self.texels[self.tmap[k].z],
                lum, col,
                self.vbuf[ self.faces[k].x ].z,
                self.vbuf[ self.faces[k].y ].z,
                self.vbuf[ self.faces[k].z ].z,
            )

            # Draw Edge

            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0),
            #    ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0),
            #    b'#', ivec( 0, 255, 255 )
            #)
            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0),
            #    ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0),
            #    b'#', ivec( 0, 255, 255 )
            #)
            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0),
            #    ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0),
            #    b'#', ivec( 0, 255, 255 )
            #)

            # Draw Vertex
            
            #col = ivec(0, 255, 255)
            #point( ivec( <int>self.vbuf[self.faces[k].x].x, <int>self.vbuf[self.faces[k].x].y, 0 ), b'A', col )
            #point( ivec( <int>self.vbuf[self.faces[k].y].x, <int>self.vbuf[self.faces[k].y].y, 0 ), b'B', col )
            #point( ivec( <int>self.vbuf[self.faces[k].z].x, <int>self.vbuf[self.faces[k].z].y, 0 ), b'C', col )

        # draw the light
        cdef vec temp
        temp.x, temp.y, temp.z = l.x, l.y, l.z
        project_point(temp, h/2, h/4)
        point( ivec(
            <int>temp.x,
            <int>temp.y,
            <int>temp.z
        ), b"#", ivec(0, 255, 255) )

        #show()

        self.i += 1
        

    #Load OBJ File (python)
    def __init__(self, FILENAME, _vcount=256):
        
        cdef int i=0
        cdef int vcount = <int>_vcount
        cdef int fcount = vcount * 4

        self.points = <vec*> calloc( vcount, sizeof(vec));
        self.normals = <vec*> calloc( fcount, sizeof(vec));
        self.texels = <vec*> calloc( fcount, sizeof(vec));
        self.faces = <ivec*> calloc( fcount, sizeof(ivec));
        self.ftex = <bool*> calloc( fcount, sizeof(bool));
        self.fmat = <int*> calloc( fcount, sizeof(int));
        self.nmap = <int*> calloc( fcount, sizeof(int));
        self.tmap = <ivec*> calloc( fcount, sizeof(ivec));

        self.vbuf = <vec*> calloc( self.vcount, sizeof(vec));
        self.dbuf = <float*> calloc( self.fcount, sizeof(float));
        self.dbuf_idx = <int*> calloc( self.fcount, sizeof(int));


        cdef int _pi = 0, _fi = 0, _ni = 0, _mi = 0, _ti = 0, _texi = 0
        mtlfile = ""
        with open(FILENAME) as f:
            for i_ in f.readlines():
                try:
                    if i_.startswith('vn'):
                        data = i_[3:].strip().replace(' ',',')
                        #_normals.append( eval(data) )
                        x, y, z = eval(data)
                        self.normals[_ni] = vec(<float>x, <float>y, <float>z)
                        _ni += 1

                    elif i_.startswith('vt'):
                        data = i_[3:].strip().replace(' ',',')
                        #_points.append( eval(data) )
                        x, y = eval(data)
                        self.texels[_ti] = vec(<float>x, <float>y, 0)
                        _ti += 1
    
                    elif i_.startswith('v'):
                        data = i_[2:].strip().replace(' ',',')
                        #_points.append( eval(data) )
                        x, y, z = eval(data)
                        self.points[_pi] = vec(<float>x, <float>y, <float>z)
                        _pi += 1
    

                    elif i_.startswith('f'): 
                        data = i_[2:].strip().replace(' ',',').replace('/',',').replace(',,',',')
                        data = [j_-1 for j_ in eval(data)]
                        #print(data)
                        #_faces.append( data[::2] )
                        #_nmap.append(data[1])

                        assert len(data) >= 6, RuntimeError("No Normals in OBJ File")
                        #assert len(data) == 9, RuntimeError("No Tex Coords in OBJ File")

                        if len(data) == 6: # [ v//n ] * 3
                            self.nmap[_fi] = data[1]
                            self.faces[_fi] = ivec( data[0], data[2], data[4] )
                        if len(data) == 9: # [ v/t/n ] * 3
                            self.nmap[_fi] = data[2]
                            self.tmap[_fi] = ivec( data[1], data[4], data[7] )
                            self.faces[_fi] = ivec( data[0], data[3], data[6] )

                        self.fmat[_fi] = _mi
                        _fi += 1

                    elif i_.startswith("mtllib"):
                        mtlfile = i_[7:]

                    elif i_.startswith("usemtl"):
                        data = i_[6:]
                        _mi += 1
                        
                        x, y, z, _texi = self.load_mtl( FILENAME, mtlfile, data, _texi, _mi)
                        self.pallete[_mi] = ivec(x, y, z)

                except Exception as e:
                    print( "in ", i_, ",", "(",FILENAME, mtlfile,")")
                    raise e
        

        self.vcount = <int>_pi
        self.fcount = <int>_fi

        self.pos = vec(0, 0, 6)
        self.rot = vec(0, 0, 0)
        
        #printf("ftex: [ ")
        #for i from 0 <= i <= _mi by 1:
        #    printf("%d ", self.ftex[i])
        #printf("]\n")

        printf("loaded model: %d verts, %d tris\n", self.vcount, self.fcount)
        
    
    def load_mtl(self, ofile, file, mat, texid, matid):
        
        with open( os.path.join(os.path.dirname(ofile) ,file.strip() ) ) as f:
            f = f.read()
        
        f = f[f.find(mat):]
        f = f[f.find("Kd"):]
        f = f[3:f.find("\n")]
        f = eval(f.replace(' ', ','))
        r, g, b = f
        r *= 255; g*= 255; b*= 255

        with open( os.path.join(os.path.dirname(ofile) ,file.strip() ) ) as f:
            f = f.read()
        
        f = f[f.find(mat):]
        f = f[f.find("map_Kd"):]
        f = f[7:f.find("\n")]

        if not f:
            self.ftex[matid] = 0
            return int(r), int(g), int(b), texid

        self.ftex[matid] = 1
        
        import numpy as np
        from PIL import Image

        #print(repr(f))
        f = np.array( Image.open( os.path.join(os.path.dirname(ofile) , f ) ) )
        
        for i in range( f.shape[0] ):
            for j in range( f.shape[1] ):
                tex_buffer[texid*TEXSIZE*TEXSIZE + j*TEXSIZE + i] = ivec( f[i, j, 0], f[i, j, 1], f[i, j, 2] )
        
        return int(r), int(g), int(b), texid+1

screen_w, screen_h = pg.size()

cpdef main():
    m = Mesh( sys.argv[1] )
    input('[press enter to continue]')
    m.rot.x = rad(90)
    m.pos.y = 4
    m.pos.z = 0

    try:
        while True:
            t = time()
        
            #m.rot.y += 1/FPS * 0.5
    
            # for mouse controls
            p = pg.position()
            cam_rot.x =  (p[1]/screen_h - 0.5) * 3.14 * 3
            cam_rot.y =  (p[0]/screen_w - 0.5) * 3.14 * 3
            #light.x =  (p[1]/screen_h - 0.5) * 200
            #light.y =  (p[0]/screen_w - 0.5) * 200
        
            clear()
            m.render()
            show()
        
            t = time()-t
            print(f"{t*1_000_000:.2f} µs {t*1000:.4f} ms {1/t:.3f} fps")
            sleep( max( 0, 1/FPS-t ) )
    
    finally:
        free(screen)
        free(color)

if len(sys.argv) > 1:
    main()

