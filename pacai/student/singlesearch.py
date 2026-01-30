"""
In this file, you will implement code relating to simple single-agent searches.
"""

import random
import typing

import pacai.agents.searchproblem
import pacai.core.agent
import pacai.core.agentaction
import pacai.core.board
import pacai.core.gamestate
import pacai.core.search
import pacai.pacman.board
import pacai.search.common
import pacai.search.food
import pacai.search.position

import pacai.util.containers
import pacai.core.search

def depth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements depth first search (DFS).
    This means that it will search the deepest nodes in the search tree first.
    See: https://en.wikipedia.org/wiki/Depth-first_search .
    """

    fringe = pacai.util.containers.Stack()
    visited = set()

    start = problem.get_starting_node()
    fringe.push((start, []))

    while not fringe.is_empty():
        state, path = fringe.pop()

        if problem.is_goal_node(state):
            return pacai.core.search.SearchSolution(path, len(path))

        if state in visited:
            continue

        visited.add(state)

        for successor, action, cost in problem.get_successor_nodes(state):
            if successor not in visited:
                fringe.push((successor, path + [action]))

    return pacai.core.search.SearchSolution([], 0)

def breadth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements breadth first search (BFS).
    This means that it will search nodes based on what level in search tree they appear.
    See: https://en.wikipedia.org/wiki/Breadth-first_search .
    """

    fringe = Queue()
    visited = set()

    start = problem.get_starting_node()
    fringe.push((start, []))
    visited.add(start)

    while not fringe.is_empty():
        state, path = fringe.pop()

        if problem.is_goal_node(state):
            return SearchSolution(path, len(path))

        for successor, action, cost in problem.get_successor_nodes(state):
            if successor not in visited:
                visited.add(successor)
                fringe.push((successor, path + [action]))

    return SearchSolution([], 0)

def uniform_cost_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements uniform cost search (UCS).
    This means that it will search nodes with a lower total cost first.
    See: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Practical_optimizations_and_infinite_graphs .
    """

    fringe = PriorityQueue()
    visited = dict()

    start = problem.get_starting_node()
    fringe.push((start, [], 0), 0)

    while not fringe.is_empty():
        state, path, cost = fringe.pop()

        if state in visited and visited[state] <= cost:
            continue

        visited[state] = cost

        if problem.is_goal_node(state):
            return SearchSolution(path, cost)

        for successor, action, step_cost in problem.get_successor_nodes(state):
            new_cost = cost + step_cost
            fringe.push((successor, path + [action], new_cost), new_cost)

    return SearchSolution([], 0)

def astar_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements A* search (pronounced "A Star search").
    This means that it will search nodes with a lower combined cost and heuristic first.
    See: https://en.wikipedia.org/wiki/A*_search_algorithm .
    """

    fringe = PriorityQueue()
    visited = dict()

    start = problem.get_starting_node()
    start_h = heuristic(start, problem)
    fringe.push((start, [], 0), start_h)

    while not fringe.is_empty():
        state, path, cost = fringe.pop()

        if state in visited and visited[state] <= cost:
            continue

        visited[state] = cost

        if problem.is_goal_node(state):
            return SearchSolution(path, cost)

        for successor, action, step_cost in problem.get_successor_nodes(state):
            new_cost = cost + step_cost
            priority = new_cost + heuristic(successor, problem)
            fringe.push((successor, path + [action], new_cost), priority)

    return SearchSolution([], 0)

class CornersSearchNode(pacai.core.search.SearchNode):
    """
    A search node the can be used to represent the corners search problem.

    You get to implement this search node however you want.
    """

    def __init__(self, position, visited_corners):
        """ Construct a search node to help search for corners. """
        self.position = position
        self.visited_corners = visited_corners

    def __eq__(self, other):
        return (
            isinstance(other, CornersSearchNode)
            and self.position == other.position
            and self.visited_corners == other.visited_corners
        )

    def __hash__(self):
        return hash((self.position, self.visited_corners))

class CornersSearchProblem(pacai.core.search.SearchProblem[CornersSearchNode]):
    """
    A search problem for touching the four different corners in a board.

    You may assume that very board is surrounded by walls (e.g., (0, 0) is a wall),
    and that the position diagonally inside from the walled corner is the location we are looking for.
    For example, if we had a square board that was 10x10, then we would be looking for the following corners:
     - (1, 1) -- North-West / Upper Left
     - (1, 8) -- North-East / Upper Right
     - (8, 1) -- South-West / Lower Left
     - (8, 8) -- South-East / Lower Right
    """

    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        self.walls = game_state.get_walls()
        self.starting_position = game_state.get_pacman_position()

        top = self.walls.get_height() - 2
        right = self.walls.get_width() - 2

        self.corners = (
            (1, 1),
            (1, top),
            (right, 1),
            (right, top)
        )

    def get_starting_node(self) -> CornersSearchNode:
        visited = tuple(
            self.starting_position == corner
            for corner in self.corners
        )
        return CornersSearchNode(self.starting_position, visited)

    def is_goal_node(self, node: CornersSearchNode) -> bool:
        return all(node.visited_corners)

    def get_successor_nodes(self, node: CornersSearchNode) -> list[pacai.core.search.SuccessorInfo]:
        successors = []
        x, y = node.position

        for direction in [
            Directions.NORTH,
            Directions.EAST,
            Directions.SOUTH,
            Directions.WEST
        ]:
            dx, dy = Directions.direction_to_vector(direction)
            next_x, next_y = int(x + dx), int(y + dy)

            if not self.walls[next_x][next_y]:
                next_position = (next_x, next_y)

                visited = list(node.visited_corners)
                for i, corner in enumerate(self.corners):
                    if next_position == corner:
                        visited[i] = True

                successors.append(
                    (CornersSearchNode(next_position, tuple(visited)),
                     direction,
                     1)
                )

        return successors

def corners_heuristic(node: CornersSearchNode, problem: CornersSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for CornersSearchProblem.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """
    position = node.position
    visited = node.visited_corners
    corners = problem.corners

    unvisited = [
        corner
        for visited_flag, corner in zip(visited, corners)
        if not visited_flag
    ]

    if not unvisited:
        return 0

    return max(manhattan(position, corner) for corner in unvisited)
    
def food_heuristic(node: pacai.search.food.FoodSearchNode, problem: pacai.search.food.FoodSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for the FoodSearchProblem.
    """
    position = node.position
    food_list = node.food.as_list()

    if not food_list:
        return 0

    distances = [manhattan(position, food) for food in food_list]
    return max(distances)

class ClosestDotSearchAgent(pacai.agents.searchproblem.GreedySubproblemSearchAgent):
    """
    Search for a path to all the food by greedily searching for the next closest food again and again
    (util we have reached all the food).

    This agent is left to you to fill out.
    But make sure to take your time and think.
    The final solution is quite simple if you take your time to understand everything up until this point and leverage
    pacai.agents.searchproblem.GreedySubproblemSearchAgent and pacai.student.problem.AnyMarkerSearchProblem.
    pacai.agents.searchproblem.GreedySubproblemSearchAgent is already implemented,
    but you should take some time to understand it.
    pacai.student.problem.AnyMarkerSearchProblem (below in this file) has not yet been implemented,
    but is the quickest and easiest way to implement this class.

    Hint:
    Remember that you can call a parent class' `__init__()` method from a child class' `__init__()` method.
    (See pacai.student.problem.AnyMarkerSearchProblem for an example.)
    Child classes will generally always call their parent's `__init__()` method
    (if a child class does not implement `__init__()`, then the parent's `__init__()` is automatically called).
    This call does not need to be the first line in the method,
    and you can pass whatever you want to the parent's `__init__()`.
    """

    def __init__(self, **kwargs):
        super().__init__(problem_class=AnyMarkerSearchProblem, **kwargs)

class AnyMarkerSearchProblem(pacai.search.position.PositionSearchProblem):
    """
    A search problem for finding a path to any instance of the specified board marker (e.g., food, wall, power capsule).

    This search problem is just like the pacai.search.position.PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    You may modify the `__init__()` if you want, the other methods should be fine as-is.
    """

    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            target_marker: pacai.core.board.Marker = pacai.pacman.board.MARKER_PELLET,
            **kwargs: typing.Any) -> None:
        super().__init__(game_state, **kwargs)

        self.target_marker = target_marker

    def is_goal_node(self, node: pacai.search.position.PositionSearchNode) -> bool:
        x, y = node.position
        return self.walls[x][y] == self.target_marker

class ApproximateSearchAgent(pacai.core.agent.Agent):
    """
    A search agent that tries to perform an approximate search instead of an exact one.
    In other words, this agent is okay with a solution that is "good enough" and not necessarily optimal.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, state: GameState):
        """
        Example strategy:
        - Move towards the nearest food using Manhattan distance.
        - This is simple, dynamic, and fast.
        """
        pacman_position = state.get_agent_position(0)
        food_positions = state.get_food().as_list()

        if not food_positions:
            return None

        nearest_food = min(
            food_positions,
            key=lambda food: abs(food[0] - pacman_position[0]) + abs(food[1] - pacman_position[1])
        )

        legal = state.get_legal_actions(0)

        best_action = None
        min_distance = float("inf")
        for action in legal:
            successor = state.generate_successor(0, action)
            succ_pos = successor.get_agent_position(0)
            distance = abs(succ_pos[0] - nearest_food[0]) + abs(succ_pos[1] - nearest_food[1])
            if distance < min_distance:
                min_distance = distance
                best_action = action

        return best_action
