# distutils: language = c++

import os, sys
from libc.stdlib cimport calloc, free #, system
from libc.stdio cimport printf
from libc.math cimport sin, cos
from time import perf_counter as time, sleep

if "DISPLAY" in os.environ: import pyautogui as pg # to get mouse position

cdef int FPS = 12;
cdef int COLOR = 1; # BOOL : ENUM(0, 1)

if os.environ.get("FPS"):     FPS = <int>int(os.environ["FPS"])
if os.environ.get("NOCOLOR"): COLOR = 0

cdef struct vec:
    float x
    float y
    float z

cdef struct ivec:
    int x
    int y
    int z

cdef int w = <int>(os.get_terminal_size()[0])
cdef int h = <int>(os.get_terminal_size()[1])

printf("\nsize: %d x %d\n", w, h)

cdef char* screen = <char*> calloc( w*h, sizeof(char) );
cdef ivec* color = <ivec*> calloc( w*h, sizeof(ivec) );

cdef void clear():
    cdef int i;
    for i from 0 <= i < w*h by 1:
        screen[i] = b" "
        color[i] = ivec(255, 255, 255)


cdef void show():
    cdef int i;
    for i from 0 <= i < w*h by 1:
        if COLOR==1 and ( color[i].x != 255 or color[i].y != 255 or color[i].z != 255 ):
            printf("\x1b[38;2;%d;%d;%dm%c\x1b[0m",
                color[i].x,
                color[i].y,
                color[i].z,
                screen[i]
            )
        else:
            printf("%c", screen[i])
    printf('\n')

cdef void point(ivec p, char c, ivec col):
    if 0 < p.y*w + p.x and p.y*w + p.x < h*w:
        screen[p.y*w + p.x] = c
        color[ p.y*w + p.x] = col

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

cdef void _flat_bot_tri(ivec p1, ivec p2, ivec p3, char c, ivec col):

    if p2.y != p3.y: printf("NOT FLAT BOTTOM TRIANGLE"); return
    if p1.y == p2.y: line(p1, p2, c, col) ; line(p2, p3, c, col); return

    cdef float m1, c1, m2, c2
    cdef int i, j

    m1 = (p2.x-p1.x)/(p2.y-p1.y)
    c1 = p1.x - m1 * p1.y

    m2 = (p3.x - p1.x)/(p3.y - p1.y)
    c2 = p1.x - m2 * p1.y

    for i from p1.y <= i <= p2.y by 1:
        if (i*m1+c1) < (i*m2+c2):
            for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                point( ivec(j, i, 0), c, col)
        else:
            for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                point( ivec(j, i, 0), c, col)

cdef void _flat_top_tri(ivec p1, ivec p2, ivec p3, char c, ivec col):

    if p2.y != p1.y: printf("NOT FLAT TOP TRIANGLE"); return
    if p2.y == p3.y: line( p1, p2, c, col ) ; line( p2, p3, c, col ); return

    cdef float m1, c1, m2, c2
    cdef int i, j

    m1 = (p1.x - p3.x)/(p1.y - p3.y)
    c1 = p3.x - m1 * p3.y

    m2 = (p3.x - p2.x)/(p3.y - p2.y)
    c2 = p3.x - m2* p3.y

    for i from p1.y <= i <= p3.y by 1:
        if (i*m1+c1) < (i*m2+c2):
            for j from <int>(i*m1+c1) <= j <= <int>(i*m2+c2) by 1:
                point( ivec(j, i, 0), c, col)
        else:
            for j from <int>(i*m2+c2) <= j <= <int>(i*m1+c1) by 1:
                point( ivec(j, i, 0), c, col)


cdef void triangle(ivec p1, ivec p2, ivec p3, char ch, ivec col):

    cdef int[3] xi = [p1.x, p2.x, p3.x]
    cdef int[3] yi = [p1.y, p2.y, p3.y]

    cdef float m, c
    cdef int i, j = 0

    for j from 1<= j < 3 by 1:
        for i from 1<= i < 3 by 1:
            if yi[i-1] > yi[i]:
                yi[i-1], yi[i] = yi[i], yi[i-1]
                xi[i-1], xi[i] = xi[i], xi[i-1]

    p1.x, p2.x, p3.x = xi[0], xi[1], xi[2]
    p1.y, p2.y, p3.y = yi[0], yi[1], yi[2]
    cdef int x4

    if   p1.y == p2.y: _flat_top_tri(p1, p2, p3, ch, col)
    elif p2.y == p3.y: _flat_bot_tri(p1, p2, p3, ch, col)
    
    else:
        m = (p1.x - p3.x)/(p1.y - p3.y)
        c = p3.x - m * p3.y

        x4 = <int>(m * p2.y + c)

        _flat_bot_tri(p1, p2, ivec(x4, p2.y, 0), ch, col)
        _flat_top_tri(p2, ivec(x4, p2.y, 0), p3, ch, col)


cdef void quad(ivec p1, ivec p2, ivec p3, ivec p4, char c, ivec col):
    triangle(p1, p2, p3, c, col)
    triangle(p2, p3, p4, c, col)


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

cdef void rotate_point(vec& p_, vec r):
    
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

    mat_mul(Rx, 3, 3, p , 3, 1, P1)
    mat_mul(Ry, 3, 3, P1, 3, 1, P2)
    mat_mul(Rz, 3, 3, P2, 3, 1, p )

    p_.x, p_.y, p_.z = p[0], p[1], p[2]

cdef void move_point(vec& p, float x, float y, float z):
    p.x += x
    p.y += y
    p.z += z

cdef vec project_point(vec& p, float sx, float sy):

    cdef float distance = 5.0
    cdef float z = 1/(distance - p.z)
    p.x = p.x * z * sx + w/2
    p.y = p.y * z * sy + h/2
    p.z = z



cdef int color_char(ivec& c):

    cdef int m
    m = ( c.x if c.x > c.y else c.y )
    m = ( m    if m    > c.z else c.z )

    c.x = <int> ( 255/m * c.x )
    c.y = <int> ( 255/m * c.y )
    c.z = <int> ( 255/m * c.z )

    return m

clear()


#cpdef class Mesh:
cdef class Mesh:
    cdef vec points[512];
    cdef vec normals[1024];
    cdef ivec faces[1024];
    cdef int fmat[1024];
    cdef int nmap[1024];

    cdef vec vbuf[512];
    cdef float dbuf[1024];
    cdef int dbuf_idx[1024];

    cdef int vcount
    cdef int fcount

    cdef int i
    
    cdef public vec pos;
    cdef public vec rot;
    

    cdef vec apply_transform(self, vec& p):
        
        rotate_point(p, self.rot)
        return p

    cdef char chars[37]

    cdef void project_points(self):
        
        cdef int j
        cdef vec p
        cdef int l = ( w if w < h else h ) * 4
        for j from 0<= j < self.vcount by 1:

            p = self.points[j]

            self.apply_transform(p)
            
            project_point(p, l, l/2)

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
        cdef vec n
        cdef vec l = vec(3, 5, 5)
        cdef float lum
        cdef ivec col
        clear()
        
        for j from 0 <= j < self.fcount by 1:
            
            k = self.dbuf_idx[j]
            n = self.normals[ self.nmap[k] ]
            l = vec(3, 5, 5)
            
            self.apply_transform(n)
            self.apply_transform(l)

            lum = ( n.x * l.x + n.y * l.y + n.z * l.z )
            lum = lum if lum > 0 else 0
            c = self.chars[<int>(lum*lum/2)]

            # Draw face
            
            if self.fmat[k] == 1:
                col.x, col.y, col.z = 255, 255, 255

            elif self.fmat[k] == 2:
                col.x, col.y, col.z = 255, 64, 64

            triangle(
                ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0),
                ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0),
                ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0),
                #b'.'
                c, col
            )

            # Draw Edge

            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0)
            #    ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0
            #    b'#', ivec( 0, 255, 255 )
            #)
            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].y ].x, <int>self.vbuf[ self.faces[k].y ].y, 0)
            #    ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0
            #    b'#', ivec( 0, 255, 255 )
            #)
            #line(
            #    ivec( <int>self.vbuf[ self.faces[k].x ].x, <int>self.vbuf[ self.faces[k].x ].y, 0)
            #    ivec( <int>self.vbuf[ self.faces[k].z ].x, <int>self.vbuf[ self.faces[k].z ].y, 0
            #    b'#', ivec( 0, 255, 255 )
            #)               
            # Draw Vertex
            
            #col = ivec(0, 255, 255)
            #point( ivec( <int>self.vbuf[self.faces[k].x].x, <int>self.vbuf[self.faces[k].x].y, 0 ), b'@', col )
            #point( ivec( <int>self.vbuf[self.faces[k].y].x, <int>self.vbuf[self.faces[k].y].y, 0 ), b'@', col ); col = ivec(0, 192, 255)
            #point( ivec( <int>self.vbuf[self.faces[k].z].x, <int>self.vbuf[self.faces[k].z].y, 0 ), b'@', col )


        show()

        self.i += 1
        

    #Load OBJ File (python)
    def __init__(self, FILENAME):
        
        cdef int i
        for i from 0<=i<37 by 1: self.chars[i] = ".:`\'-,;~_!\"?c\\^<>|=sr1Jo*(C)utia3zLvey75jST{lx}IfY]qp9n0G62Vk8UXhZ4bgdPEKA$wQm&#HDR@WNBM"[i]

        cdef int _pi = 0, _fi = 0, _ni = 0, _mi = 0

        with open(FILENAME) as f:
            for i_ in f.readlines():

                if i_.startswith('vn'):
                    data = i_[3:].strip().replace(' ',',')
                    #_normals.append( eval(data) )
                    x, y, z = eval(data)
                    self.normals[_ni] = vec(<float>x, <float>y, <float>z)
                    _ni += 1

                elif i_.startswith('v'):
                    data = i_[2:].strip().replace(' ',',')
                    #_points.append( eval(data) )
                    x, y, z = eval(data)
                    self.points[_pi] = vec(<float>x, <float>y, <float>z)
                    _pi += 1
    

                elif i_.startswith('f'): 
                    data = i_[2:].strip().replace(' ',',').replace('/',',').replace(',,',',')
                    data = [j_-1 for j_ in eval(data)]
                    #_faces.append( data[::2] )
                    #_nmap.append(data[1])

                    assert len(data) == 6, RuntimeError("No Normals in OBJ File")
    
                    self.nmap[_fi] = data[1]
                    self.faces[_fi] = ivec( data[0], data[2], data[4] )
    
                    self.fmat[_fi] = _mi
                    _fi += 1

                elif i_.startswith("usemtl"):
                    data = i_.lstrip("usemtl ")
                    #materials.append(data)
                    _mi += 1
        

        self.vcount = <int>_pi
        self.fcount = <int>_fi

        printf("loaded model: %d verts, %d tris\n", self.vcount, self.fcount)

cpdef main():
    m = Mesh( sys.argv[1] )
    input('[press enter to continue]')
    m.rot.x = rad(100)
    while True:
        t = time()
        
        m.rot.y += 1/FPS * 0.5

        # for mouse controls
        #p = pg.position()
        #m.rot.x = -p[1]/100
        #m.rot.y = -p[0]/100

        m.render()
    
        t = time()-t
        print(f"{t*1_000_000} µs {t*1000:.4f} ms {1/t:.3f} fps")
        sleep( max( 0, 1/FPS-t ) )

main()


free(screen)
free(color)
#"""
