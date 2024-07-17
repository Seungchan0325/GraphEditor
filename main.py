import heapq
from math import sqrt
from random import randint
from collections import deque

from tkinter import *
from tkinter import ttk
import tkinter
from tkinter.scrolledtext import ScrolledText
from tkinter.tix import NoteBook

def set_text_bgcolor(widget, line, color):
    widget.tag_add(f'{line}', f'{line}.0', f'{line}.end')
    widget.tag_config(f'{line}', background=color)

class DSU:
    def __init__(self, N):
        self.root = [i for i in range(0, N+1)]
    
    def find(self, x: int) -> int:
        if self.root[x] == x:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]
    
    def merge(self, x: int, y: int):
        x = self.find(x)
        y = self.find(y)
        self.root[y] = x

class Main(Tk):
    def __init__(self):
        super().__init__()

        self.initUI()

        # Graph
        self.N = 0
        self.edges = []
        self.nodes = [0]
        self.graph = []

        self.num_step = -1
        self.steps = []

        self.selected_node = -1
        self.cv_edges = []
        self.cv_edges_color = []

        self.SET_NODE = 0
        self.SET_EDGE = 1
        self.UNSET_NODE = 2
        self.UNSET_EDGE = 3
        self.OUTPUT_TEXT = 4

        self.r = 20

        self.update_radio()
        self.analyze()
    
    def initUI(self):
        self.title('Graph Editor')

        self.left_nodebook = ttk.Notebook(self, width=200)
        self.left_nodebook.pack(side='left', fill='y', padx=5, pady=5)

        ## Input
        self.input_frame = Frame(self)
        self.left_nodebook.add(self.input_frame, text='Input')
        
        self.btn_update = Button(self.input_frame, text='Update', command=lambda: (self.read_graph(), self.update_radio()))
        self.btn_update.pack(side='top', fill='x')

        self.is_direction = IntVar()
        self.cb_is_direction = Checkbutton(self.input_frame, text='Direction', variable=self.is_direction)
        self.cb_is_direction.pack(side='top', anchor='w')
        
        self.is_weight = IntVar()
        self.cb_is_weight = Checkbutton(self.input_frame, text='Weight', variable=self.is_weight)
        self.cb_is_weight.pack(side='top', anchor='w')

        self.lb_node_cnt = Label(self.input_frame, text='Node Count')
        self.lb_node_cnt.pack(side='top', anchor='center')

        self.tb_node_cnt = Entry(self.input_frame)
        self.tb_node_cnt.pack(side='top', fill='x')

        self.lb_edges = Label(self.input_frame, text = 'Graph Edges')
        self.lb_edges.pack(side='top', anchor='center')

        self.tb_edges = ScrolledText(self.input_frame)
        self.tb_edges.pack(side='top', fill='both', expand=True)

        ## Output
        self.output_frame = Frame(self)
        self.left_nodebook.add(self.output_frame, text='Output')

        self.buttons_frame = Frame(self.output_frame)
        self.buttons_frame.pack(side='top', fill='x')

        self.btn_run = Button(self.buttons_frame, text='Run', command=self.run_algorithm)
        self.btn_run.pack(side='left', expand=True, fill='x')

        self.btn_step = Button(self.buttons_frame, text='Step', command=self.step_algorithm)
        self.btn_step.pack(side='left', expand=True, fill='x')

        self.btn_stop = Button(self.buttons_frame, text='Stop', command=self.stop_algorithm)
        self.btn_stop.pack(side='left', expand=True, fill='x')

        self.algorithm = IntVar()

        self.algorithms = [('DFS', self.dfs, True, (0, 0)),
                           ('BFS', self.bfs, True, (0, 0)),
                           ('Bellman Ford', self.bellman_ford, True, (0, 1)),
                           ('Dijkstra', self.dijkstra, True, (0, 1)),
                           ('Minimum Spanning Tree', self.minimum_spanning_tree, True, (-1, 1)),
                           ('Maximum Spanning Tree', self.maximum_spanning_tree, True, (-1, 1)),
                           ('Eulerian Circuit', self.eulerian_circuit, True, (-1, 0)),
                           ('Eulerian Trail', self.eulerian_trail, True, (-1, 0)),
                           ('Scc', self.scc, False, (1, 0)),
                           ('Cut Vertices', self.cut_vertices, False, (-1, 0)),
                           ('Bridge', self.bridge, False, (-1, 0)),
                        #    ('Max Flow', self.max_flow, True, (1, 1))
                           ]

        self.rad_algorithm: list[Radiobutton] = []
        for i, algorithm in enumerate(self.algorithms):
            rad = Radiobutton(self.output_frame, text=algorithm[0], variable=self.algorithm, value=i, command=self.algorithm_changed)
            rad.pack(side='top', anchor='w')
            self.rad_algorithm.append(rad)

        self.lb_start_node = Label(self.output_frame, text='Start Node')
        self.lb_start_node.pack(side='top')
        
        self.tb_start_node = Entry(self.output_frame)
        self.tb_start_node.pack(side='top', fill='x')

        self.lb_output = Label(self.output_frame, text='Output')
        self.lb_output.pack(side='top')

        self.tb_output = ScrolledText(self.output_frame, height=10, state='disabled')
        self.tb_output.pack(side='top', fill='both', expand=True)

        # Canvas
        self.canvas = Canvas(self, relief='solid', bd=1)
        self.canvas.bind('<Button-1>', self.select)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.drop)
        self.canvas.bind('<Configure>', self.resize)
        self.canvas.pack(side='right', expand=True, fill='both', padx=5, pady=5)

    def read_graph(self):
        try:
            N = self.tb_node_cnt.get()

            if not N.isdecimal() or int(N) > 1000:
                raise

            N = int(N)
            if self.N < N:
                for i in range(self.N + 1, N + 1):
                    self.add_node(i)
            else:
                for i in range(self.N, N, -1):
                    cx, cy, circle, txt = self.nodes.pop()
                    self.canvas.delete(circle)
                    self.canvas.delete(txt)

            self.N = N
            self.tb_node_cnt.configure(bg='white')
        except:
            self.tb_node_cnt.configure(bg='red')
            N = 0

        # read edges
        edges: list[tuple[int, int, int]] = []
        lines = int(self.tb_edges.index('end-1c').split('.')[0])
        for i in range(1, lines+1):
            try:
                uvw = self.tb_edges.get(f'{i}.0', f'{i}.end').split()
                if not (2 <= len(uvw) <= 3):
                    raise

                u = uvw[0]
                v = uvw[1]
                w = '0'

                if self.is_weight.get():
                    if len(uvw) != 3:
                        raise
                    w = uvw[2]

                if not u.isdecimal() or not v.isdecimal() or not (w.isdecimal() or (w[0] == '-' and w[1:].isdecimal)):
                    raise

                u = int(u)
                v = int(v)
                w = int(w)

                if not (1 <= u <= N) or not (1 <= v <= N) or u == v:
                    raise

                edges.append((u, v, w))
                set_text_bgcolor(self.tb_edges, i, 'white')
            except:
                set_text_bgcolor(self.tb_edges, i, 'red')

        self.graph = [[] for i in range(N+1)]
        for u, v, w in edges:
            if not self.is_direction.get():
                self.graph[v].append((u, w))
            self.graph[u].append((v, w))
        
        self.edges = edges
        self.cv_edges_color = [['black' for i in range(N+1)] for i in range(N+1)]
        self.draw_edges()

        self.stop_algorithm()
        self.analyze()
    
    def update_radio(self):
        self.algorithm.set(0)
        for i, algorithm in enumerate(self.algorithms):
            name, func, steppable, requires = algorithm

            disabled = False

            if requires[0] == 1:
                if not self.is_direction.get(): disabled = True
            elif requires[0] == -1:
                if self.is_direction.get(): disabled = True
            
            if requires[1] == 1:
                if not self.is_weight.get(): disabled = True
            elif requires[1] == -1:
                if self.is_weight.get(): disabled = True
            
            if disabled: self.rad_algorithm[i].configure(state='disabled')
            else: self.rad_algorithm[i].configure(state='normal')

    def resize(self, event):
        for i in range(1, self.N+1):
            x, y = self.nodes[i][:2]
            x = max(min(x, self.canvas.winfo_width()), 0)
            y = max(min(y, self.canvas.winfo_height()), 0)

            dx = x - self.nodes[i][0]
            dy = y - self.nodes[i][1]

            self.nodes[i][0] = x
            self.nodes[i][1] = y

            self.canvas.move(self.nodes[i][2], dx, dy)
            self.canvas.move(self.nodes[i][3], dx, dy)

        self.draw_edges()
    
    def add_node(self, idx):
        cx = randint(0, self.canvas.winfo_width())
        cy = randint(0, self.canvas.winfo_height())
        r = self.r

        circle = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill='gray', width=3)
        txt = self.canvas.create_text(cx, cy, text=f'{idx}')

        self.nodes.append([cx, cy, circle, txt])

    def draw_edges(self):
        for line, txt in self.cv_edges:
            self.canvas.delete(line)
            if txt != -1:
                self.canvas.delete(txt)
        
        self.cv_edges = []

        for [u, v, w] in self.edges:
            x0, y0 = self.nodes[u][:2]
            x1, y1 = self.nodes[v][:2]

            r = self.r
            if x1 == x0:
                if y0 <= y1:
                    y0 += r
                    y1 -= r
                else:
                    y0 -= r
                    y1 += r
            else:
                m = (y1 - y0) / (x1 - x0)
                if x0 < x1: r = -r
                x0 -= r / sqrt(m*m+1)
                y0 -= m * r / sqrt(m*m+1)

                x1 += r / sqrt(m*m+1)
                y1 += m * r / sqrt(m*m+1)

            if self.is_direction.get():
                line = self.canvas.create_line(x0, y0, x1, y1, width=5, arrow=LAST, fill=self.cv_edges_color[u][v])
            else:
                line = self.canvas.create_line(x0, y0, x1, y1, width=5, fill=self.cv_edges_color[u][v])
            self.canvas.lower(line)

            txt = -1
            if self.is_weight.get():
                x = (x0 + x1) // 2 - 10
                y = (y0 + y1) // 2 - 10
                txt = self.canvas.create_text(x, y, text=f'{w}')
            self.cv_edges.append([line, txt])

    def select(self, event):
        x, y = event.x, event.y
        for i in range(1, self.N+1):
            cx, cy, cycle, txt = self.nodes[i]
            r = self.r

            if (cx - x)**2 + (cy - y)**2 <= r**2:
                self.selected_node = i
                break

    def drop(self, event):
        self.selected_node = -1

    def drag(self, event):
        if self.selected_node != -1:
            x, y = event.x, event.y
            
            x = max(min(x, self.canvas.winfo_width()), 0)
            y = max(min(y, self.canvas.winfo_height()), 0)
            
            i = self.selected_node
            
            dx = x - self.nodes[i][0]
            dy = y - self.nodes[i][1]

            self.nodes[i][0] = x
            self.nodes[i][1] = y

            self.canvas.move(self.nodes[i][2], dx, dy)
            self.canvas.move(self.nodes[i][3], dx, dy)

            self.draw_edges()
    
    def run_algorithm(self):
        self.stop_algorithm()
        self.analyze()

        name, func, steppable, requires = self.algorithms[self.algorithm.get()]

        self.steps = func()
        self.num_step = 0
        for i in self.steps:
            self.step_algorithm()
        self.num_step = -1
    
    def step_algorithm(self):
        name, func, steppable, requires = self.algorithms[self.algorithm.get()]

        if self.num_step == -1:
            self.stop_algorithm()
            self.analyze()
            self.steps = func()
            self.num_step = 0
        
        if self.num_step == len(self.steps):
            if len(self.steps) == 0:
                self.num_step = -1
            return
        
        self.tb_output.configure(state='normal')

        step = self.steps[self.num_step]

        if step[0] == self.SET_NODE:
            u = step[1]
            self.canvas.itemconfigure(self.nodes[u][2], fill='blue')
            self.tb_output.insert(END, f'Node {u}\n')
        elif step[0] == self.SET_EDGE:
            u = step[1]
            v = step[2]
            self.cv_edges_color[u][v] = 'red'
            if not self.is_direction.get():
                self.cv_edges_color[v][u] = 'red'
            self.tb_output.insert(END, f'Edge ({step[1]}, {step[2]})\n')
        elif step[0] == self.UNSET_NODE:
            u = step[1]
            self.canvas.itemconfigure(self.nodes[u][2], fill='gray')
            self.tb_output.insert(END, f'Node {u}\n')
        elif step[0] == self.UNSET_EDGE:
            u = step[1]
            v = step[2]
            self.cv_edges_color[u][v] = 'black'
            if not self.is_direction.get():
                self.cv_edges_color[v][u] = 'black'
            self.tb_output.insert(END, f'Edge ({step[1]}, {step[2]})\n')
        elif step[0] == self.OUTPUT_TEXT:
            self.tb_output.insert(END, step[1])
        self.draw_edges()

        self.num_step += 1
        self.tb_output.configure(state='disabled')

    def stop_algorithm(self):
        self.cv_edges_color = [['black' for i in range(self.N+1)] for i in range(self.N+1)]
        self.draw_edges()
        for cx, cy, circle, txt in self.nodes[1:]:
            self.canvas.itemconfigure(circle, fill='gray')

        self.analyze()
        self.num_step = -1
    
    def algorithm_changed(self):
        name, func, steppable, requires = self.algorithms[self.algorithm.get()]

        self.tb_start_node.configure(background='white')

        self.stop_algorithm()
        if steppable:
            self.btn_step.configure(state='normal')
            self.btn_stop.configure(state='normal')
        else:
            self.btn_step.configure(state='disabled')
            self.btn_stop.configure(state='disabled')
    
    def analyze(self):
        self.tb_output.configure(state='normal')
        self.tb_output.delete('1.0', END)

        self.tb_output.insert(END, f'Is Tree: {self.is_tree()}\n')
        self.tb_output.insert(END, f'Is Forest: {self.is_forest()}\n')
        self.tb_output.insert(END, f'Is Connected Graph: {self.count_connecting_components() == 1}\n')
        self.tb_output.insert(END, f'Is DAG: {self.is_dag()}\n')
        self.tb_output.insert(END, f'Connecting Component: {self.count_connecting_components()}\n')

        self.tb_output.configure(state='disabled')

    def is_tree(self) -> bool:
        if self.is_direction.get():
            return False

        if self.count_connecting_components() != 1:
            return False

        N = self.N
        edges = self.edges
        dsu = DSU(N)

        for u, v, w in edges:
            if dsu.find(u) == dsu.find(v):
                return False
            dsu.merge(u, v)

        for i in range(2, N+1):
            if dsu.find(i-1) != dsu.find(i):
                return False

        return True

    def is_forest(self) -> bool:
        if self.is_direction.get():
            return False
        N = self.N
        edges = self.edges
        dsu = DSU(N)

        for u, v, w in edges:
            if dsu.find(u) == dsu.find(v):
                return False
            dsu.merge(u, v)

        return True

    def is_dag(self):
        if not self.is_direction.get():
            return False
        
        N = self.N
        edges = self.edges
        graph = self.graph

        indeg = [0 for i in range(0, N+1)]

        for u, v, w in edges:
            indeg[v] += 1
        
        q = deque()
        for i in range(1, N+1):
            if indeg[i] == 0:
                q.append(i)
        
        for i in range(N):
            if not q:
                return False

            u = q.popleft()
            for v, w in graph[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        
        return True

    def count_connecting_components(self):
        N = self.N
        edges = self.edges
        dsu = DSU(N)

        for u, v, w in edges:
            dsu.merge(u, v)

        cnt = 0
        visited = [False for i in range(N+1)]
        for i in range(1, N+1):
            if visited[dsu.find(i)] == False:
                cnt += 1
                visited[dsu.find(i)] = True

        return cnt

    def get_start_node(self):
        n = self.tb_start_node.get()
        if not n.isdecimal():
            self.tb_start_node.configure(background='red')
            return -1
        
        n = int(n)
        if n < 1 or n > self.N:
            self.tb_start_node.configure(background='red')
            return -1

        self.tb_start_node.configure(background='white')
        return n

    def dfs(self):
        N = self.N
        graph = self.graph.copy()
        visited = [False for i in range(N+1)]
        steps = []
        def dfs(self: Main, u):
            steps.append((self.SET_NODE, u))
            visited[u] = True
            for v, w in graph[u]:
                if visited[v]: continue
                steps.append((self.SET_EDGE, u, v))
                dfs(self, v)
        
        start = self.get_start_node()
        if start == -1:
            return []
        
        dfs(self, start)
        
        return steps

    def bfs(self):
        N = self.N
        visited = [False for i in range(N+1)]
        graph = self.graph.copy()
        steps = []

        start = self.get_start_node()
        if start == -1: return []

        q = deque()
        q.append(start)
        visited[start] = True

        while q:
            u = q.popleft()
            steps.append((self.SET_NODE, u))

            for v, w in graph[u]:
                if visited[v]: continue
                q.append(v)
                visited[v] = True
                steps.append((self.SET_EDGE, u, v))
        
        return steps

    def bellman_ford(self):
        N = self.N
        edges = self.edges.copy()
        INF = int(1e18)
        distance = [INF for i in range(N+1)]

        start = self.get_start_node()
        if start == -1: return []

        distance[start] = 0
        steps = []

        for i in range(N):
            for u, v, w in edges:
                if distance[u] != INF and distance[v] > distance[u] + w:
                    steps.append((self.SET_EDGE, u, v))
                    distance[v] = distance[u] + w
                    if i == N-1:
                        steps.append((self.OUTPUT_TEXT, "Cycle Detected\n"))
                        return steps

                if not self.is_direction.get():
                    steps.append((self.SET_EDGE, v, u))
                    u, v = v, u
                    if distance[u] != INF and distance[v] > distance[u] + w:
                        distance[v] = distance[u] + w
                        if i == N-1:
                            steps.append((self.OUTPUT_TEXT, "Cycle Detected\n"))
                            return steps
        
        output = ''
        for i in range(1, N+1):
            if distance[i] == INF:
                output += f'{i}: INF\n'
            else:
                output += f'{i}: {distance[i]}\n'

        steps.append((self.OUTPUT_TEXT, output))
        return steps

    def dijkstra(self):
        N = self.N
        graph = self.graph.copy()
        INF = int(1e18)
        distance = [INF for i in range(N+1)]

        start = self.get_start_node()

        if start == -1: return []

        steps = []
 
        pq = [(0, start, -1)]
        distance[start] = 0

        while pq:
            now, u, par = heapq.heappop(pq)

            if now > distance[u]: continue

            if par != -1:
                steps.append((self.SET_EDGE, par, u))

            for v, w in graph[u]:
                d = now + w
                if d < distance[v]:
                    distance[v] = d
                    heapq.heappush(pq, (d, v, u))
        
        output = ''
        for i in range(1, N+1):
            if distance[i] == INF:
                output += f'{i}: INF\n'
            else:
                output += f'{i}: {distance[i]}\n'
        
        steps.append((self.OUTPUT_TEXT, output))
        return steps

    def minimum_spanning_tree(self):
        N = self.N
        edges = self.edges.copy()
        graph = self.graph.copy()

        dsu = DSU(N)
        edges.sort(key=lambda x:x[2])

        steps = []

        sum = 0

        for u, v, w in edges:
            if dsu.find(u) == dsu.find(v): continue
            sum += w
            dsu.merge(u, v)
            steps.append((self.SET_EDGE, u, v))
        
        steps.append((self.OUTPUT_TEXT, f'MST: {sum}'))
        
        return steps

    def maximum_spanning_tree(self):
        N = self.N
        edges = self.edges.copy()
        graph = self.graph.copy()

        dsu = DSU(N)
        edges.sort(key=lambda x:x[2], reverse=True)

        steps = []

        sum = 0

        for u, v, w in edges:
            if dsu.find(u) == dsu.find(v): continue
            sum += w
            dsu.merge(u, v)
            steps.append((self.SET_EDGE, u, v))
        
        steps.append((self.OUTPUT_TEXT, f'MST: {sum}'))
        
        return steps

    def eulerian_circuit(self):
        N = self.N
        edges = self.edges.copy()
        graph = self.graph.copy()

        adjmat = [[0 for i in range(N+1)] for j in range(N+1)]

        for u, v, w in edges:
            adjmat[u][v] += 1
            if not self.is_direction.get():
                adjmat[v][u] += 1

        steps = []
        path = []
        
        for i in range(1, N+1):
            if sum(adjmat[i]) % 2 == 1:
                steps.append((self.OUTPUT_TEXT, f"{sum(adjmat[i])}Can't find eulerian circuit"))
                return steps
            
        start = self.get_start_node()
        if start == -1: return []

        def dfs(self: Main, u):
            path.append(f'{u}')
            for v in range(1, N+1):
                while adjmat[u][v]:
                    adjmat[u][v] -= 1
                    if not self.is_direction.get():
                        adjmat[v][u] -= 1
                    steps.append((self.SET_EDGE, u, v))
                    dfs(self, v)
        
        dfs(self, start)

        steps.append((self.OUTPUT_TEXT, '->'.join(path)))

        return steps

    def eulerian_trail(self):
        N = self.N
        edges = self.edges.copy()
        graph = self.graph.copy()

        adjmat = [[0 for i in range(N+1)] for j in range(N+1)]

        for u, v, w in edges:
            adjmat[u][v] += 1
            if not self.is_direction.get():
                adjmat[v][u] += 1

        steps = []
        path = []
        
        cnt = 0
        for i in range(1, N+1):
            if sum(adjmat[i]) % 2 == 1:
                cnt += 1
            
        start = self.get_start_node()
        if start == -1: return []
        
        if (cnt != 0 and cnt != 2) or (cnt == 2 and sum(adjmat[start]) % 2 != 1):
            steps.append((self.OUTPUT_TEXT, f"Can't find eulerian circuit"))
            return steps

        def dfs(self: Main, u):
            path.append(f'{u}')
            for v in range(1, N+1):
                while adjmat[u][v]:
                    adjmat[u][v] -= 1
                    if not self.is_direction.get():
                        adjmat[v][u] -= 1
                    steps.append((self.SET_EDGE, u, v))
                    dfs(self, v)
        
        dfs(self, start)

        steps.append((self.OUTPUT_TEXT, '->'.join(path)))

        return steps

    def scc(self):
        N = self.N
        edges = self.edges.copy()
        graph = self.graph.copy()

        graph_rev = [[] for i in range(N+1)]

        for u, v, w in edges:
            graph_rev[v].append((u, w))

        steps = []
        order = []
        SCC = [-1 for i in range(N+1)]
        chk = [False for i in range(N+1)]

        def dfs(self, u):
            chk[u] = True
            for v, w in graph[u]:
                if not chk[v]:
                    dfs(self, v)
            order.append(u)
        
        def dfs_rev(self: Main, u, num):
            chk[u] = True
            SCC[u] = num
            for v, w in graph_rev[u]:
                steps.append((self.SET_EDGE, v, u))

                if not chk[v]:
                    dfs_rev(self, v, num)

        for i in range(1, N+1):
            if not chk[i]:
                dfs(self, i)

        chk = [False for i in range(N+1)]
        
        order.reverse()
        num = 0
        for u in order:
            if SCC[u] == -1:
                dfs_rev(self, u, num)
                num += 1
        
        ans = [[] for i in range(num)]
        for u in range(1, N+1):
            ans[SCC[u]].append(f'{u}')
        
        ans.sort()
        
        output = ''
        for i, _scc in enumerate(ans):
            output += f'{i+1}: ' + ', '.join(_scc) + '\n'
        
        steps.append((self.OUTPUT_TEXT, output))

        return steps

    def cut_vertices(self):
        N = self.N
        edges = self.edges.copy()
        graph = self.graph.copy()

        order = [0 for i in range(N+1)]
        low = [0 for i in range(N+1)]
        par = [0 for i in range(N+1)]
        is_ap = [False for i in range(N+1)]
        global t
        t = 0

        def dfs(self: Main, u):
            global t
            t += 1
            order[u] = t
            low[u] = t
            
            sub = 0
            for v, w in graph[u]:
                if par[u] == v: continue

                if not order[v]:
                    par[v] = u
                    sub += 1
                    dfs(self, v)

                    if not par[u] and sub > 1: is_ap[u] = True
                    if par[u] and low[v] >= order[u]: is_ap[u] = True
                    low[u] = min(low[u], low[v])
                else:
                    low[u] = min(low[u], order[v])

        for u in range(1, N+1):
            if not order[u]:
                dfs(self, u)

        steps = []

        for i in range(1, N+1):
            if is_ap[i]:
                steps.append((self.SET_NODE, i))

        return steps

    def bridge(self):
        N = self.N
        edges = self.edges.copy()
        graph = self.graph.copy()

        order = [0 for i in range(N+1)]
        low = [0 for i in range(N+1)]
        par = [0 for i in range(N+1)]
        global t
        t = 0

        steps = []

        def dfs(self: Main, u):
            global t
            t += 1
            order[u] = t
            low[u] = t

            for v, w in graph[u]:
                if par[u] == v: continue

                if not order[v]:
                    par[v] = u
                    dfs(self, v)

                    if low[v] > order[u]:
                        steps.append((self.SET_EDGE, min(u, v), max(u, v)))

                    low[u] = min(low[u], low[v])
                else:
                    low[u] = min(low[u], order[v])
        
        for i in range(1, N+1):
            if not order[i]:
                dfs(self, i)

        return steps

    def max_flow(self):
        N = self.N
        edges = self.edges
        graph = self.graph

if __name__ == '__main__':
    main = Main()
    main.mainloop()
