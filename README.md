# Graph Editor
이산수학 그래프를 시각화하고 다양한 알고리즘을 시각화합니다.

## 기능
1. 그래프를 그려줍니다.
2. 다양한 그래프 알고리즘을 실행합니다.

### 지원하는 알고리즘
1. DFS
2. BFS
3. Bellman ford
4. Dijkstra
5. Maximum spanning tree
6. Minimum spanning tree
7. Eulerian circuit
8. Eulerian trail
9. Strongly connected component
10. Cut vertices (단절점 찾기)
11. Bridge (단절선 찾기)

## 사용법

### Input 창
+ Update 버튼: 입력을 적용한다.
+ Direction, Weight 버튼: 방향의 유무, 가중치의 유무를 설정한다.
+ Node Count: 노드의 개수를 입력한다.
+ Graph Edges: 간선의 정보를 입력받는다. 각 간선은 줄 단위로 구분 되어야 한다. 각 줄에는 간선이 잇는 두 정점과 만약 가중치가 있다면 가중치가 주어져야한다. 따라서 (u v [w]) 형식이어야 한다. 만약 방향그래프라면 u에서 v로 가는 간선이다.

### Output 창
+ Run 버튼: 알고리즘을 실행한다.
+ Step 버튼: 알고리즘을 한 단계 실행한다.
+ Stop 버튼: 알고리즘 실행 결과를 초기화한다.
+ 알고리즘 선택 라디오 버튼: 알고리즘을 선택한다.
+ Start Node: 시작 정점을 필요로하는 알고리즘이 사용하는 시작 정점
+ Output: 그래프 분석 결과와 알고리즘의 출력이 나오는 텍스트 박스

### 파이썬 버전
+ Python 3.9.13
+ Tcl/Tk version 8.6.12

## Run
```python main.py```