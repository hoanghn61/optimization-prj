from ortools.sat.python import cp_model

model = cp_model.CpModel()

finp = open("input.txt","r")

def read():
    while(True):
        xx = finp.readline()
        if len(xx)!=1:
            return list(map(int, xx.split()))

#N task and M workers
nTask, nWorker = map(int, finp.readline().split())

task_duration = []
t_wl = [] #task _ worker list: a list of available workers for a task
for t in range(nTask):
    xx = read()[:-1]
    task_duration.append(xx[0])
    t_wl.append([x-1 for x in xx[1:]])

#s[i]: the time that worker i can start his work 
s = read()

K = int(finp.readline()[:-1])

ff = []
#ff (finish first): ff[i] mean that task ff[i][0] must finish before the start of task ff[i][1]
for i in range(K):
    xx = read()
    xx[0]-=1
    xx[1]-=1
    ff.append(xx)

#c: c[i][j] is the cost of worker j to finish task i
c = []
for t in range(nTask):
    c.append(read())

#tw (task-worker): if tw[i][j] == 1 mean that worker j do task i
tw = [[0 for _ in range(nWorker)] for __ in range(nTask)]
for t in range(nTask):
    for w in range(nWorker):
        tw[t][w] = model.NewBoolVar(f"tw_{t}_{w}")
        if w not in t_wl[t]: #if worker w is not allowed to do task t
            model.AddHint(tw[t][w], 0)

#each task only done ONCE
for t in range(nTask):
    model.AddExactlyOne(tw[t])

#max_time that can finish all work
max_time = sum(task_duration) + min(s)

from collections import namedtuple
taskData = [namedtuple("info",["begin","end","interval","worker"]) for _ in range(nTask)]

for t in range(nTask):
    #task t begin time:
    taskData[t].begin = model.NewIntVar(0,max_time,f"taskData_{t}_begin")
    #task t end time:
    taskData[t].end = model.NewIntVar(0,max_time,f"taskData_{t}_end")
    #task t interval(begin time, duration, end time = begin time + duration)
    taskData[t].interval = model.NewIntervalVar(taskData[t].begin, task_duration[t], taskData[t].end, f"taskData_{t}_interval")
    #worker that do task t:
    taskData[t].worker = model.NewIntVar(0,nWorker,f"worker_for_task_{t}")
    model.Add(taskData[t].worker == sum(tw[t][w]*w for w in range(nWorker)))
    

for t in range(nTask):
    for w in range(nWorker):
        #if worker w do task t then the start time of that task must be larger than s[w] (s[w] is the starting time of worker w)
        model.Add(taskData[t].begin>=s[w]).OnlyEnforceIf(tw[t][w])

for i in range(K):
    task1 = ff[i][0]
    task2 = ff[i][1]
    #task1 must be finished before task2
    model.Add(taskData[task1].end <= taskData[task2].begin)

#same_worker[t1][t2] = True means that task t1 and task t2 are done by same worker.
same_worker = [[0 for t2 in range(nTask)] for t1 in range(nTask)]
for t1 in range(nTask):
    for t2 in range(t1+1,nTask):
        same_worker[t1][t2] = model.NewBoolVar(f"same_worker_{t1}_{t2}")
        model.AddMaxEquality(same_worker[t1][t2], [tw[t1][w]+tw[t2][w]-1 for w in range(nWorker)])

for t1 in range(nTask):
    for t2 in range(t1+1,nTask):
        #for given two ranges [x1,x2], [y1,y2], they are overlap if max(x1,y1) < min(x2,y2)
        max_x1_y1 = model.NewIntVar(0,max_time,f"max_x1_y1_{t1}_{t2}")
        min_x2_y2 = model.NewIntVar(0,max_time,f"min_x2_y2_{t1}_{t2}") 
        model.AddMaxEquality(max_x1_y1,[taskData[t1].begin,taskData[t2].begin])     # max_x1_y1 = max(x1,y1)
        model.AddMinEquality(min_x2_y2,[taskData[t2].end,taskData[t2].end])         # min_x2_y2 = min(x2,y2)
        # if task t1 and t2 are done by same worker, they must not be overlapped.
        # max(x1,y1) >= min(x2,y2) if same_worker[t1][t2] = True
        model.Add(max_x1_y1>=min_x2_y2).OnlyEnforceIf(same_worker[t1][t2])          

max_cost = sum(c[t][w] for t in range(nTask) for w in range(nWorker))
min_cost = model.NewIntVar(0, max_cost, 'min_cost')
#min cost = highest cost paid for a worker
model.AddMaxEquality(min_cost, [sum(c[t][w]*tw[t][w] for t in range(nTask)) for w in range(nWorker)])        

min_time = model.NewIntVar(0, max_time, 'min_time')
model.AddMaxEquality(min_time, [taskData[t].end for t in range(nTask)])
model.Minimize(min_time*max_time+min_cost)

fout = open("1.out","w")

def w(x="",end = "\n"):
    fout.write(format(x))
    fout.write(end)

global_min_time = max_time
global_min_cost = max_cost
global_taskData = [namedtuple("info",["begin","end","worker"]) for _ in range(nTask)]

#this callback function is called whenever our code found a FEASIBLE case
class SolutionChecker(cp_model.CpSolverSolutionCallback):

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)

    # iterate through all FEASIBLE solution to find min_time fisrt and then min_cost
    def on_solution_callback(self):
        global global_min_time
        global global_min_cost
        global global_taskData
        if (global_min_time > self.Value(min_time)) or (global_min_time == self.Value(min_time) and global_min_cost > self.Value(min_cost)):
            global_min_time = self.Value(min_time)
            global_min_cost = self.Value(min_cost)
            for t in range(nTask):
                global_taskData[t].begin = self.Value(taskData[t].begin)
                global_taskData[t].end = self.Value(taskData[t].end)
                global_taskData[t].worker = self.Value(taskData[t].worker)
        return

solver = cp_model.CpSolver()

solver.parameters.enumerate_all_solutions = True
callback = SolutionChecker()
status = solver.Solve(model,callback)

#status = solver.Solve(model)

if status in [cp_model.UNKNOWN]:
    print("Time out.")
    exit(0)

if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print("No solution.")
    exit(0)

print(f"The minimun time is: {global_min_time}")
print(f"With that time, the minimun biggest cost for a worker is: {global_min_cost}")

print("Detail of all tasks: \n")

for t in range(nTask):
    print(f"Task {t+1}:")
    print(f"Begin at: {global_taskData[t].begin}")
    print(f"End at:   {global_taskData[t].end}")
    print(f"Finish by worker {global_taskData[t].worker+1}")
    print()
