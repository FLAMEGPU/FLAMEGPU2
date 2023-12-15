import pytest
from unittest import TestCase
from pyflamegpu import *

ID_AGENT_COUNT = 1025;
ID_OFFSET = 29034;
ID_GAP1 = 3;
ID_GAP2 = 5;

class InitGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(10);
        vertices = graph.vertices();
        for i in range(1, 11):
            vertex = vertices[i];
            vertex.setPropertyFloat("vertex_float", i);
            vertex.setPropertyDouble("vertex_double2", 0, i + 11);
            vertex.setPropertyDouble("vertex_double2", 1, i + 21);  # Redundant for Py
            vertex.setPropertyArrayInt("vertex_int3", [i + 1, i + 2, i + 3]);

        graph.setEdgeCount(20);
        edges = graph.edges();
        for i in range(10):
            edge = edges[(i / 2) + 1,((i + 2) % 10) + 1];
            edge.setPropertyInt("edge_int", i + 70);
            edge.setPropertyDouble("edge_double2", 0, i + 61);
            edge.setPropertyDouble("edge_double2", 1, i + 51);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [ i + 41, i + 42, i + 43]);

        for i in range(10, 20):
            edge = edges[(i / 2) + 1,((i + 6) % 10) + 1];
            edge.setPropertyInt("edge_int", i + 70);
            edge.setPropertyDouble("edge_double2", 0, i + 61);
            edge.setPropertyDouble("edge_double2", 1, i + 51);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [ i + 41, i + 42, i + 43]);


# Init's same as InitGraph, however fills the vertices/edges with zero
class InitGraph2(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(10);
        vertices = graph.vertices();
        for i in range(1, 11):
            vertex = vertices[i];
            vertex.setPropertyFloat("vertex_float", 0);
            vertex.setPropertyDouble("vertex_double2", 0, 0);
            vertex.setPropertyDouble("vertex_double2", 1, 0);  # Redundant for Py
            vertex.setPropertyArrayInt("vertex_int3", [0, 0, 0]);

        graph.setEdgeCount(20);
        edges = graph.edges();
        for i in range(10):
            edge = edges[(i % 10) + 1, 1];
            edge.setPropertyInt("edge_int", 0);
            edge.setPropertyDouble("edge_double2", 0, 0);
            edge.setPropertyDouble("edge_double2", 1, 0);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [0, 0, 0]);

        for i in range(10, 20):
            edge = edges[(i % 10) + 1, 3];
            edge.setPropertyInt("edge_int", 0);
            edge.setPropertyDouble("edge_double2", 0, 0);
            edge.setPropertyDouble("edge_double2", 1, 0);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [0, 0, 0]);

# Alternate version to InitGraph
class InitGraph3(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(30);
        vertices = graph.vertices();
        for i in range(1, 31):
            vertex = vertices[i];
            vertex.setPropertyFloat("vertex_float", i);
            vertex.setPropertyDouble("vertex_double2", 0, i + 11);
            vertex.setPropertyDouble("vertex_double2", 1, i + 21);  # Redundant for Py
            vertex.setPropertyArrayInt("vertex_int3", [i + 1, i + 2, i + 3]);

        graph.setEdgeCount(60);
        edges = graph.edges();
        for i in range(30):
            edge = edges[(i / 2) + 1, ((i + 2) % 30) + 1];
            edge.setPropertyInt("edge_int", i + 70);
            edge.setPropertyDouble("edge_double2", 0, i + 61);
            edge.setPropertyDouble("edge_double2", 1, i + 51);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [i + 41, i + 52, i + 43]);

        for i in range(30, 60):
            edge = edges[(i / 2) + 1, ((i + 18) % 30) + 1];
            edge.setPropertyInt("edge_int", i + 70);
            edge.setPropertyDouble("edge_double2", 0, i + 61);
            edge.setPropertyDouble("edge_double2", 1, i + 51);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [i + 41, i + 52, i + 43]);

# Set graph to same data as InitGraph, it assumes vertice/edge counts are correct
class SetGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        vertices = graph.vertices();
        for i in range(1, 11):
            vertex = vertices[i];
            vertex.setPropertyFloat("vertex_float", i);
            vertex.setPropertyDouble("vertex_double2", 0, i + 11);
            vertex.setPropertyDouble("vertex_double2", 1, i + 21);  # Redundant for Py
            vertex.setPropertyArrayInt("vertex_int3", [i + 1, i + 2, i + 3]);

        edges = graph.edges();
        for i in range(10):
            edge = edges[(i % 10) + 1, 1];
            edge.setDestinationVertexID(2);
            edge.setSourceVertexID(i + 1);
            edge.setPropertyInt("edge_int", i + 70);
            edge.setPropertyDouble("edge_double2", 0, i + 61);
            edge.setPropertyDouble("edge_double2", 1, i + 51);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [i + 41, i + 42, i + 43]);

        for i in range(10, 20):
            edge = edges[(i % 10) + 1, 3];
            edge.setSourceDestinationVertexID(i - 9, 4);
            edge.setPropertyInt("edge_int", i + 70);
            edge.setPropertyDouble("edge_double2", 0, i + 61);
            edge.setPropertyDouble("edge_double2", 1, i + 51);  # Redundant for Py
            edge.setPropertyArrayFloat("edge_float3", [i + 41, i + 42, i + 43]);

class HostCheckSetGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        vertices = graph.vertices();
        # Vertices
        assert graph.getVertexCount() == 10;
        for i in range(1, 11):
            vertex = vertices[i];
            assert vertex.getPropertyFloat("vertex_float") == i;
            assert vertex.getPropertyDouble("vertex_double2", 0) == i + 11;
            # assert vertex.getPropertyDouble("vertex_double2", 1) == i + 21;  # Redundant for Py
            assert vertex.getPropertyArrayInt("vertex_int3") == (i + 1, i + 2, i + 3);

        # Edges
        edges = graph.edges();
        assert graph.getEdgeCount() == 20;
        for i in range(10):
            edge = edges[i + 1, 2];
            assert edge.getPropertyInt("edge_int") == i + 70;
            assert edge.getPropertyDouble("edge_double2", 0) == i + 61;
            # assert edge.getPropertyDouble("edge_double2", 1) == i + 51;  # Redundant for Py
            assert edge.getPropertyArrayFloat("edge_float3") == (i + 41, i + 42, i + 43);

        for i in range(10, 20):
            edge = edges[i - 9, 4];
            assert edge.getPropertyInt("edge_int") == i + 70;
            assert edge.getPropertyDouble("edge_double2", 0) == i + 61;
            # assert edge.getPropertyDouble("edge_double2", 1) == si + 51;  # Redundant for Py
            assert edge.getPropertyArrayFloat("edge_float3") == (i + 41, i + 42, i + 43);
class HostCheckGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        vertices = graph.vertices();
        # Vertices
        assert graph.getVertexCount() == 10;
        for i in range(1, 11):
            vertex = vertices[i];
            assert vertex.getPropertyFloat("vertex_float") == i;
            assert vertex.getPropertyDouble("vertex_double2", 0) == i + 11;
            # assert vertex.getPropertyDouble("vertex_double2", 1) == i + 21;  # Redundant for Py
            assert vertex.getPropertyArrayInt("vertex_int3") == (i + 1, i + 2, i + 3);

        # Edges
        edges = graph.edges();
        assert graph.getEdgeCount() == 20;
        for i in range(10):
            edge = edges[(i / 2) + 1, ((i + 2) % 10) + 1];
            assert edge.getPropertyInt("edge_int") == i + 70;
            assert edge.getPropertyDouble("edge_double2", 0) == i + 61;
            # assert edge.getPropertyDouble("edge_double2", 1) == i + 51;  # Redundant for Py
            assert edge.getPropertyArrayFloat("edge_float3") == (i + 41, i + 42, i + 43);

        for i in range(10, 20):
            edge = edges[(i / 2) + 1, ((i + 6) % 10) + 1];
            assert edge.getPropertyInt("edge_int") == i + 70;
            assert edge.getPropertyDouble("edge_double2", 0) == i + 61;
            # assert edge.getPropertyDouble("edge_double2", 1) == si + 51;  # Redundant for Py
            assert edge.getPropertyArrayFloat("edge_float3") == (i + 41, i + 42, i + 43);

# Equivalent version to HostCheckGraph but for InitGraph3
class HostCheckGraph3(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        vertices = graph.vertices();
        # Vertices
        assert graph.getVertexCount() == 30;
        for i in range(1, 31):
            vertex = vertices[i];
            assert vertex.getPropertyFloat("vertex_float") == i;
            assert vertex.getPropertyDouble("vertex_double2", 0) == i + 11;
            # assert vertex.getPropertyDouble("vertex_double2", 1) == i + 21;  # Redundant for Py
            assert vertex.getPropertyArrayInt("vertex_int3") == (i + 1, i + 2, i + 3);

        # Edges
        edges = graph.edges();
        assert graph.getEdgeCount() == 60;
        for i in range(30):
            edge = edges[(i / 2) + 1, ((i + 2) % 30) + 1];
            assert edge.getSourceVertexID() == int(i / 2) + 1;
            assert edge.getDestinationVertexID() == ((i + 2) % 30)+1;
            assert edge.getPropertyInt("edge_int") == i + 70;
            assert edge.getPropertyDouble("edge_double2", 0) == i + 61;
            # assert edge.getPropertyDouble("edge_double2", 1) == i + 51;  # Redundant for Py
            assert edge.getPropertyArrayFloat("edge_float3") == (i + 41, i + 52, i + 43);

        for i in range(30, 60):
            edge = edges[(i / 2) + 1, ((i + 18) % 30) + 1];
            assert edge.getPropertyInt("edge_int") == i + 70;
            assert edge.getPropertyDouble("edge_double2", 0) == i + 61;
            # assert edge.getPropertyDouble("edge_double2", 1) == i + 51;  # Redundant for Py
            assert edge.getPropertyArrayFloat("edge_float3") == (i + 41, i + 52, i + 43);

class HostException(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
    
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphName exception
            FLAMEGPU.environment.getDirectedGraph("does not exist");
        assert e.value.type() == "InvalidGraphName"
        graph = FLAMEGPU.environment.getDirectedGraph("graph");

        vertices = graph.vertices();
        edges = graph.edges();
    
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException exception
            vertices[1]
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException exception
            edges[1,2]
        assert e.value.type() == "OutOfBoundsException"
        
        graph.setVertexCount(1);
        graph.setEdgeCount(1);
        
        vertex = vertices[1];
        edge = edges[1, 1];

        # Name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.setPropertyFloat("does not exist", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     vertex.setPropertyArrayDouble("does not exist", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.setPropertyArrayInt("does not exist", []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.setPropertyInt("does not exist", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     edge.setPropertyArrayDouble("does not exist", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.setPropertyArrayFloat("does not exist", []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.getPropertyFloat("does not exist");
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     vertex.getPropertyArrayDouble("does not exist", 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.getPropertyArrayInt("does not exist");
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.getPropertyInt("does not exist");
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     edge.getPropertyArrayDouble("does not exist", 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.getPropertyArrayFloat("does not exist");
        assert e.value.type() == "InvalidGraphProperty"

        # Type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.setPropertyUInt("vertex_float", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     vertex.setPropertyArrayUInt("vertex_double2", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.setPropertyArrayUInt("vertex_int3", []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.setPropertyUInt("edge_int", 0);
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     edge.setPropertyArrayUInt("edge_double2", 0, 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.setPropertyArrayUInt("edge_float3", []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.getPropertyUInt("vertex_float");
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     vertex.getPropertyArrayUInt("vertex_double2", 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.getPropertyArrayUInt("vertex_int3");
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.getPropertyUInt("edge_int");
        assert e.value.type() == "InvalidGraphProperty"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
        #     edge.getPropertyArrayUInt("edge_double2", 0);
        # assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.getPropertyArrayUInt("edge_float3");
        assert e.value.type() == "InvalidGraphProperty"
        
        # Length
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.setPropertyInt("vertex_int3", 0);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.setPropertyArrayInt("vertex_int3", []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.setPropertyArrayInt("vertex_int3", [1, 2]);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.setPropertyFloat("edge_float3", 0);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.setPropertyArrayFloat("edge_float3", []);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.setPropertyArrayFloat("edge_float3", [1, 2]);
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            vertex.getPropertyInt("vertex_int3");
        assert e.value.type() == "InvalidGraphProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidGraphProperty exception
            edge.getPropertyFloat("edge_float3");
        assert e.value.type() == "InvalidGraphProperty"


        # Array Index
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # vertex.setPropertyArrayDouble("vertex_double2", 3, 0);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # edge.setPropertyArrayDouble("edge_double2", 3, 0);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          #  vertex.getPropertyArrayDouble("vertex_double2", 3);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          #   edge.getPropertyArrayDouble("edge_double2", 3);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # vertex.setPropertyArrayDouble("vertex_double2", 3, 0);
        # assert e.value.type() == "OutOfBoundsException"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
          # edge.setPropertyArrayDouble("edge_double2", 3, 0);
        # assert e.value.type() == "OutOfBoundsException"

        # Vertex/Edge Index
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            vertices[0];
        assert e.value.type() == "IDOutOfBounds"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            vertex.setID(0)
        assert e.value.type() == "IDOutOfBounds"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            edges[0, 1];
        assert e.value.type() == "IDOutOfBounds"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            edges[1, 0];
        assert e.value.type() == "IDOutOfBounds"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            edge.setSourceVertexID(0)
        assert e.value.type() == "IDOutOfBounds"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            edge.setDestinationVertexID(0)
        assert e.value.type() == "IDOutOfBounds"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            edge.setSourceDestinationVertexID(0, 1)
        assert e.value.type() == "IDOutOfBounds"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::OutOfBoundsException 
            edge.setSourceDestinationVertexID(1, 0)
        assert e.value.type() == "IDOutOfBounds"
        
class HostDeviceCheckGraph(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent").getPopulationData();
        for i in range(20):
            if i < 10:
                assert agent[i].getVariableID("vertex_id") == i + 1;
                assert agent[i].getVariableFloat("vertex_float") == i+ 1;
                assert agent[i].getVariableArrayDouble("vertex_double2") == (i + 12, i + 22);
                assert agent[i].getVariableID("edge_dest") == ((i + 2) % 10) + 1;
            else:
                assert agent[i].getVariableID("edge_dest") == ((i + 6) % 10) + 1;
            # Edges
            assert agent[i].getVariableID("edge_source") == int(i / 2) + 1;
            assert agent[i].getVariableInt("edge_int") == i + 70;
            assert agent[i].getVariableArrayDouble("edge_double2") == (i + 61, i + 51);

class HostDeviceCheckGraph3(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent").getPopulationData();
        for i in range(60):
            if i < 30:
                assert agent[i].getVariableID("vertex_id") == i + 1;
                assert agent[i].getVariableFloat("vertex_float") == i + 1;
                assert agent[i].getVariableArrayDouble("vertex_double2") == (i + 12, i + 22);
                assert agent[i].getVariableID("edge_dest") == ((i + 2) % 30) + 1;
            else:
                assert agent[i].getVariableID("edge_dest") == ((i + 18) % 30) + 1;
            # Edges
            assert agent[i].getVariableID("edge_source") == int(i / 2) + 1;
            assert agent[i].getVariableInt("edge_int") == i + 70;
            assert agent[i].getVariableArrayDouble("edge_double2") == (i + 61, i + 51);

class always_exit(pyflamegpu.HostCondition):
    def run(self, FLAMEGPU):
        return pyflamegpu.EXIT;

class HostTestEdgesOut(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(5);
        graph.setEdgeCount(15);
        vertices = graph.vertices();
        edges = graph.edges();
        for i in range(5):
            vertices[i + 1];
            for j in range(i + 1):
                edge = edges[j + 1, i + 1];
                edge.setPropertyID("src_copy", j);
                
class HostTestEdgesIn(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(5);
        graph.setEdgeCount(15);
        vertices = graph.vertices();
        edges = graph.edges();
        for i in range(5):
            vertices[i + 1];
            for j in range(i + 1):
                edge = edges[i + 1, j + 1];
                edge.setPropertyID("dest_copy", j);

class InitGraph_ContiguousIDs(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        OFFSET = ID_OFFSET;
        graph.setVertexCount(ID_AGENT_COUNT);
        graph.setEdgeCount(ID_AGENT_COUNT);
        vertices = graph.vertices();
        edges = graph.edges();
        for i in range(ID_AGENT_COUNT):
            vertex = vertices[OFFSET + i];
            vertex.setPropertyUInt("vertex_index", i);
            vertex.setPropertyUInt("vertex_ID", OFFSET + i);
            # Test does not care about edges, but add some so that it generates properly
            edges[OFFSET + i, OFFSET + ((i + 5) * 3) % ID_AGENT_COUNT];

class InitGraph_NonContiguousIDs(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        OFFSET = ID_OFFSET;
        graph.setVertexCount(ID_AGENT_COUNT);
        graph.setEdgeCount(ID_AGENT_COUNT);
        vertices = graph.vertices();
        edges = graph.edges();
        for i in range(ID_AGENT_COUNT):
            my_id = OFFSET + i + ((i * ID_GAP2) - ((i * ID_GAP1) % ID_GAP2));
            vertex = vertices[my_id];
            vertex.setPropertyUInt("vertex_index", i);
            vertex.setPropertyUInt("vertex_ID", my_id);
            # Test does not care about edges, but add some so that it generates properly
            j = ((i + 5) * 3) % ID_AGENT_COUNT;
            my_dest_id = OFFSET + j + ((j * ID_GAP2) - ((j * ID_GAP1) % ID_GAP2));
            edges[my_id, my_dest_id];

class InitGraph_InvalidEdgeSource(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # ID range > vertex count
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        OFFSET = ID_OFFSET;
        graph.setVertexCount(ID_AGENT_COUNT);
        graph.setEdgeCount(ID_AGENT_COUNT);
        vertices = graph.vertices();
        edges = graph.edges();
        for i in range(ID_AGENT_COUNT):
            my_id = OFFSET + i*5;
            vertex = vertices[my_id];
            j = i + 1 % int(ID_AGENT_COUNT/2);
            edges[my_id, OFFSET + j * 5];
            
        i = int(ID_AGENT_COUNT / 2);
        my_id = OFFSET + i*5;
        j = i + 1 % int(ID_AGENT_COUNT/2);
        edge = edges[my_id, OFFSET + j * 5];
        edge.setSourceVertexID(OFFSET - 1);
        edge.setDestinationVertexID(OFFSET);

class InitGraph_InvalidEdgeDest(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # ID range > vertex count
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        OFFSET = ID_OFFSET;
        graph.setVertexCount(ID_AGENT_COUNT);
        graph.setEdgeCount(ID_AGENT_COUNT);
        vertices = graph.vertices();
        edges = graph.edges();
        for i in range(ID_AGENT_COUNT):
            my_id = OFFSET + i*5;
            vertex = vertices[my_id];
            j = i + 1 % int(ID_AGENT_COUNT/2);
            edges[my_id, OFFSET + j * 5];

        i = int(ID_AGENT_COUNT / 2);
        my_id = OFFSET + i*5;
        j = i + 1 % int(ID_AGENT_COUNT/2);
        edge = edges[my_id, OFFSET + j * 5];
        edge.setSourceVertexID(OFFSET);
        edge.setDestinationVertexID(OFFSET - 1);

class InitGraph_NoEdges(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # ID range > vertex count
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(ID_AGENT_COUNT);
        graph.setEdgeCount(0);
        vertices = graph.vertices();
        for i in range(ID_AGENT_COUNT):
            vertex = vertices[i + 1];
            vertex.setPropertyInt("foo", i);

class InitGraph_SameSrcDest(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
        # ID range > vertex count
        graph = FLAMEGPU.environment.getDirectedGraph("graph");
        graph.setVertexCount(ID_AGENT_COUNT);
        graph.setEdgeCount(ID_AGENT_COUNT);
        vertices = graph.vertices();
        edges = graph.edges();
        for i in range(ID_AGENT_COUNT):
            vertex = vertices[i + 1];
            vertex.setPropertyInt("foo", i);
            edge = edges[i + 1, i + 1];
            edge.setPropertyUInt("id", i);

class EnvironmentDirectedGraphTest(TestCase):
    def test_HostGetResetGet(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());
        # Check the data persists
        model.newLayer().addHostFunction(HostCheckGraph().__disown__());
        # Init graph with different known data
        model.newLayer().addHostFunction(InitGraph3().__disown__());
        # Check the data persists
        model.newLayer().addHostFunction(HostCheckGraph3().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();

    def test_HostSetGet(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Init graph with junk data
        model.newLayer().addHostFunction(InitGraph2().__disown__());
        # Set the graphs data to known data
        model.newLayer().addHostFunction(SetGraph().__disown__());
        # Check the data persists
        model.newLayer().addHostFunction(HostCheckSetGraph().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();
      
    def test_HostException(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Init graph with junk data
        model.newLayer().addHostFunction(HostException().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();
    
    CopyGraphToAgent1_func = """
    FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent1, flamegpu::MessageNone, flamegpu::MessageNone) {
        if (FLAMEGPU->getID() <= 20) {
            flamegpu::DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
            if (FLAMEGPU->getID() <= 10) {
                FLAMEGPU->setVariable<flamegpu::id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
                // vertex_int3, device full array access not available, so skipped
            }
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_source", graph.getVertexID(graph.getEdgeSource(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_dest", graph.getVertexID(graph.getEdgeDestination(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
            FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
            // edge_float3, device full array access not available, so skipped
        }
        return flamegpu::ALIVE;
    }
    """
    
    CopyGraphToAgent3_func = """
    FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent3, flamegpu::MessageNone, flamegpu::MessageNone) {
        if (FLAMEGPU->getID() <= 60) {
            flamegpu::DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
            if (FLAMEGPU->getID() <= 30) {
                FLAMEGPU->setVariable<flamegpu::id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
                FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
                // vertex_int3, device full array access not available, so skipped
            }
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_source", graph.getVertexID(graph.getEdgeSource(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
            FLAMEGPU->setVariable<flamegpu::id_t>("edge_dest", graph.getVertexID(graph.getEdgeDestination(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
            FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
            // edge_float3, device full array access not available, so skipped
        }
        return flamegpu::ALIVE;
    }
    """
    
    def test_DeviceGetResetGet(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        agent = model.newAgent("agent");

        agent.newVariableID("vertex_id");
        agent.newVariableFloat("vertex_float");
        agent.newVariableArrayDouble("vertex_double2", 2);
        # agent.newVariableArrayInt("vertex_int3", 3);, device full array access not available, so skipped
        agent.newVariableID("edge_source");
        agent.newVariableID("edge_dest");
        agent.newVariableInt("edge_int");
        agent.newVariableArrayDouble("edge_double2", 2);
        # agent.newVariableArrayFloat("edge_float3", 3);, device full array access not available, so skipped
        CopyGraphToAgent1 = agent.newRTCFunction("fn1", self.CopyGraphToAgent1_func);
        CopyGraphToAgent3 = agent.newRTCFunction("fn2", self.CopyGraphToAgent3_func);

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());
        # Copy Data from Graph to Agent
        model.newLayer().addAgentFunction(CopyGraphToAgent1);
        # Check the agent data is correct persists
        model.newLayer().addHostFunction(HostDeviceCheckGraph().__disown__());
        # Init graph with different known data
        model.newLayer().addHostFunction(InitGraph3().__disown__());
        # Copy Data from Graph to Agent
        model.newLayer().addAgentFunction(CopyGraphToAgent3);
        # Check the agent data is correct persists
        model.newLayer().addHostFunction(HostDeviceCheckGraph3().__disown__());

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(agent, 60);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

    def test_SubModelSet_MasterModelHostGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");
        submodel.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);

        # Init graph with known data
        submodel.newLayer().addHostFunction(InitGraph().__disown__());
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();

        # Init graph with known data
        model.newLayer().addSubModel(sub_desc);

        # Check the data persists
        model.newLayer().addHostFunction(HostCheckGraph().__disown__());

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();

    def test_SubModelSet_MasterModelDeviceGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");

        agent = submodel.newAgent("agent");
        agent.newVariableFloat("foobar");  # Agent is not required in this model for this test

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);

        submodel.newLayer().addHostFunction(InitGraph().__disown__());

        # Copy data to agent and check in host fn
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        master_agent = model.newAgent("agent");
        master_agent.newVariableFloat("foobar");  # Agent is only used to init a population
        master_agent.newVariableID("vertex_id");
        master_agent.newVariableFloat("vertex_float");
        master_agent.newVariableArrayDouble("vertex_double2", 2);
        # master_agent.newVariableArrayInt("vertex_int3", 3);, device full array access not available, so skipped
        master_agent.newVariableID("edge_source");
        master_agent.newVariableID("edge_dest");
        master_agent.newVariableInt("edge_int");
        master_agent.newVariableArrayDouble("edge_double2", 2);
        # master_agent.newVariableArrayFloat("edge_float3", 3);, device full array access not available, so skipped

        CopyGraphToAgent1 = master_agent.newRTCFunction("fn1", self.CopyGraphToAgent1_func);

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();
        sub_desc.bindAgent("agent", "agent");

        # Init graph with known data
        model.newLayer().addSubModel(sub_desc);

        # Check the data persists
        model.newLayer().addAgentFunction(CopyGraphToAgent1);
        model.newLayer().addHostFunction(HostDeviceCheckGraph().__disown__());

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(master_agent, 20);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

    def test_MasterModelSet_SubModelHostGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");
        submodel.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);

        # Init graph with known data
        submodel.newLayer().addHostFunction(HostCheckGraph().__disown__());
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        model.newAgent("agent").newVariableFloat("foobar");  # Agent is not used in this test

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());

        # Check the data persists
        model.newLayer().addSubModel(sub_desc);

        sim = pyflamegpu.CUDASimulation(model);

        sim.step();

    def test_MasterModelSet_SubModelDeviceGet(self):
        submodel = pyflamegpu.ModelDescription("SubGraphTest");
        sub_graph = submodel.Environment().newDirectedGraph("graph");

        agent = submodel.newAgent("agent");
        agent.newVariableID("vertex_id");
        agent.newVariableFloat("vertex_float");
        agent.newVariableArrayDouble("vertex_double2", 2);
        # agent.newVariableArrayInt("vertex_int3", 3);, device full array access not available, so skipped
        agent.newVariableID("edge_source");
        agent.newVariableID("edge_dest");
        agent.newVariableInt("edge_int");
        agent.newVariableArrayDouble("edge_double2", 2);
        # agent.newVariableArrayFloat("edge_float3", 3);, device full array access not available, so skipped

        CopyGraphToAgent1 = agent.newRTCFunction("fn1", self.CopyGraphToAgent1_func);

        sub_graph.newVertexPropertyFloat("vertex_float");
        sub_graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        sub_graph.newVertexPropertyArrayInt("vertex_int3", 3);

        sub_graph.newEdgePropertyInt("edge_int");
        sub_graph.newEdgePropertyArrayDouble("edge_double2", 2);
        sub_graph.newEdgePropertyArrayFloat("edge_float3", 3);


        # Copy data to agent and check in host fn
        submodel.newLayer().addAgentFunction(CopyGraphToAgent1);
        submodel.newLayer().addHostFunction(HostDeviceCheckGraph().__disown__());
        submodel.addExitCondition(always_exit().__disown__());

        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyFloat("vertex_float");
        graph.newVertexPropertyArrayDouble("vertex_double2", 2);
        graph.newVertexPropertyArrayInt("vertex_int3", 3);

        graph.newEdgePropertyInt("edge_int");
        graph.newEdgePropertyArrayDouble("edge_double2", 2);
        graph.newEdgePropertyArrayFloat("edge_float3", 3);

        master_agent = model.newAgent("agent");
        master_agent.newVariableFloat("foobar");  # Agent is only used to init a population

        # Setup submodel
        sub_desc = model.newSubModel("sub_graph", submodel);
        sub_desc.SubEnvironment().autoMapDirectedGraphs();
        sub_desc.bindAgent("agent", "agent");

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph().__disown__());

        # Check the data persists
        model.newLayer().addSubModel(sub_desc);

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(master_agent, 20);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();
        
    EdgesOut_func = """
    FLAMEGPU_AGENT_FUNCTION(IterateEdges, flamegpu::MessageNone, flamegpu::MessageNone) {
        flamegpu::id_t src = FLAMEGPU->getIndex();
        unsigned int ct = 0;
        bool src_all_correct = true;
        auto filter = FLAMEGPU->environment.getDirectedGraph("graph").outEdges(src);
        FLAMEGPU->setVariable<int>("count2", filter.size());
        for (auto &edge : filter) {
            src_all_correct &= edge.getProperty<flamegpu::id_t>("src_copy") == src;
            FLAMEGPU->setVariable<flamegpu::id_t, 5>("dests", ct, edge.getEdgeDestination());
            ++ct;
        }
        FLAMEGPU->setVariable<int>("count", ct);
        FLAMEGPU->setVariable<int>("src_all_correct", src_all_correct ? 1 : 0);
        return flamegpu::ALIVE;
    }
    """
        
    def test_EdgesOut(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newEdgePropertyID("src_copy");

        agent = model.newAgent("agent");
        agent.newVariableArrayID("dests", 5);
        agent.newVariableInt("count");
        agent.newVariableInt("count2");
        agent.newVariableInt("src_all_correct");
        EdgesOut = agent.newRTCFunction("iterate_edges", self.EdgesOut_func);

        # Init graph with known data
        model.newLayer().addHostFunction(HostTestEdgesOut());
        model.newLayer().addAgentFunction(EdgesOut);

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(agent, 5);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        sim.getPopulationData(pop);
        k = int(0);
        for agt in pop:
            assert agt.getVariableInt("src_all_correct") == 1;
            assert agt.getVariableInt("count") == 5 - k;
            assert agt.getVariableInt("count2") == 5 - k;
            for i in range(5 - k):
                assert agt.getVariableID("dests", i) == k + i;
            k += 1;
            
    EdgesIn_func = """
    FLAMEGPU_AGENT_FUNCTION(IterateEdges, flamegpu::MessageNone, flamegpu::MessageNone) {
        flamegpu::id_t dest = FLAMEGPU->getIndex();
        unsigned int ct = 0;
        bool dest_all_correct = true;
        auto filter = FLAMEGPU->environment.getDirectedGraph("graph").inEdges(dest);
        FLAMEGPU->setVariable<int>("count2", filter.size());
        for (auto& edge : filter) {
            dest_all_correct &= edge.getProperty<flamegpu::id_t>("dest_copy") == dest;
            FLAMEGPU->setVariable<flamegpu::id_t, 5>("srcs", ct, edge.getEdgeSource());
            ++ct;
        }
        FLAMEGPU->setVariable<int>("count", ct);
        FLAMEGPU->setVariable<int>("dest_all_correct", dest_all_correct ? 1 : 0);
        return flamegpu::ALIVE;
    }
    """
    
    def test_EdgesIn(self):
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newEdgePropertyID("dest_copy");

        agent = model.newAgent("agent");
        agent.newVariableArrayID("srcs", 5);
        agent.newVariableInt("count");
        agent.newVariableInt("count2");
        agent.newVariableInt("dest_all_correct");
        EdgesIn = agent.newRTCFunction("iterate_edges", self.EdgesIn_func);

        # Init graph with known data
        model.newLayer().addHostFunction(HostTestEdgesIn());
        model.newLayer().addAgentFunction(EdgesIn);

        # Create enough agents, to copy all data from the 2nd graph init
        pop = pyflamegpu.AgentVector(agent, 5);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        sim.getPopulationData(pop);
        k = int(0);
        for agt in pop:
            assert agt.getVariableInt("dest_all_correct") == 1;
            assert agt.getVariableInt("count") == 5 - k;
            assert agt.getVariableInt("count2") == 5 - k;
            for i in range(5 - k):
                assert agt.getVariableID("srcs", i) == k + i;
            k += 1;

    CheckGraph_ContiguousIDs_func = """
    const unsigned int ID_AGENT_COUNT = 1025;
    const unsigned int ID_OFFSET = 29034;
    const unsigned int ID_GAP1 = 3;
    const unsigned int ID_GAP2 = 5;
    FLAMEGPU_AGENT_FUNCTION(CheckGraph_ContiguousIDs, flamegpu::MessageNone, flamegpu::MessageNone) {
        auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
        unsigned int my_vertex_id = ID_OFFSET + FLAMEGPU->getIndex();
        unsigned int my_vertex_index = graph.getVertexIndex(my_vertex_id);
        unsigned int my_vertex_id2 = graph.getVertexID(my_vertex_index);
        unsigned int my_vertex_id3 = graph.getVertexProperty<unsigned int>("vertex_ID", my_vertex_index);
        unsigned int my_vertex_index2 = graph.getVertexProperty<unsigned int>("vertex_index", my_vertex_index);

        if (my_vertex_id == my_vertex_id2) {
            FLAMEGPU->setVariable<unsigned int>("result1", 1);
        }
        if (my_vertex_id == my_vertex_id3) {
            FLAMEGPU->setVariable<unsigned int>("result2", 1);
        }
        if (my_vertex_index == my_vertex_index2) {
            FLAMEGPU->setVariable<unsigned int>("result3", 1);
        }
        return flamegpu::ALIVE;
    }
    """

    def test_VertexIDContiguous(self):
        # Assign vertices an ID and check they are accessible correctly via map
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyUInt("vertex_index");
        graph.newVertexPropertyUInt("vertex_ID");

        agent = model.newAgent("agent");
        agent.newVariableUInt("result1", 0);
        agent.newVariableUInt("result2", 0);
        agent.newVariableUInt("result3", 0);
        a = agent.newRTCFunction("check_graph", self.CheckGraph_ContiguousIDs_func);

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_ContiguousIDs());
        model.newLayer().addAgentFunction(a);

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        sim.getPopulationData(pop);
        for agt in pop:
            assert agt.getVariableUInt("result1") == 1
            assert agt.getVariableUInt("result2") == 1
            assert agt.getVariableUInt("result3") == 1

    CheckGraph_NonContiguousIDs_func = """
    const unsigned int ID_AGENT_COUNT = 1025;
    const unsigned int ID_OFFSET = 29034;
    const unsigned int ID_GAP1 = 3;
    const unsigned int ID_GAP2 = 5;
    FLAMEGPU_AGENT_FUNCTION(CheckGraph_NonContiguousIDs, flamegpu::MessageNone, flamegpu::MessageNone) {
        auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
        unsigned int my_vertex_id = ID_OFFSET + FLAMEGPU->getIndex() + ((FLAMEGPU->getIndex()*ID_GAP2) - ((FLAMEGPU->getIndex()*ID_GAP1) % ID_GAP2));
        unsigned int my_vertex_index = graph.getVertexIndex(my_vertex_id);
        unsigned int my_vertex_id2 = graph.getVertexID(my_vertex_index);
        unsigned int my_vertex_id3 = graph.getVertexProperty<unsigned int>("vertex_ID", my_vertex_index);
        unsigned int my_vertex_index2 = graph.getVertexProperty<unsigned int>("vertex_index", my_vertex_index);

        if (my_vertex_id == my_vertex_id2) {
            FLAMEGPU->setVariable<unsigned int>("result1", 1);
        }
        if (my_vertex_id == my_vertex_id3) {
            FLAMEGPU->setVariable<unsigned int>("result2", 1);
        }
        if (my_vertex_index == my_vertex_index2) {
            FLAMEGPU->setVariable<unsigned int>("result3", 1);
        }
        return flamegpu::ALIVE;
    }
    """
    
    def test_VertexIDNonContiguous(self):
        # Assign vertices an ID and check they are accessible correctly via map
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        graph.newVertexPropertyUInt("vertex_index");
        graph.newVertexPropertyUInt("vertex_ID");

        agent = model.newAgent("agent");
        agent.newVariableUInt("result1", 0);
        agent.newVariableUInt("result2", 0);
        agent.newVariableUInt("result3", 0);
        a = agent.newRTCFunction("check_graph", self.CheckGraph_NonContiguousIDs_func);

        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_NonContiguousIDs());
        model.newLayer().addAgentFunction(a);

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        sim.getPopulationData(pop);
        for agt in pop:
            assert agt.getVariableUInt("result1") == 1
            assert agt.getVariableUInt("result2") == 1
            assert agt.getVariableUInt("result3") == 1

    def test_InvalidEdgeSource(self):
        # Assign vertices an ID and check they are accessible correctly via map
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        agent = model.newAgent("agent");
        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_InvalidEdgeSource());

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidID 
            sim.step()
        assert e.value.type() == "InvalidID"

    def test_InvalidEdgeDest(self):
        # Assign vertices an ID and check they are accessible correctly via map
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");

        agent = model.newAgent("agent");
        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_InvalidEdgeDest());

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidID 
            sim.step()
        assert e.value.type() == "InvalidID"
    
    CheckGraphNoEdges_func = """
    FLAMEGPU_AGENT_FUNCTION(CheckGraphNoEdges, flamegpu::MessageNone, flamegpu::MessageNone) {
        auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
        // Can still access vertex data
        const unsigned int id = graph.getVertexID(FLAMEGPU->getIndex());
        const unsigned int id_minus_1 = graph.getVertexProperty<int>("foo", FLAMEGPU->getIndex());
        if (id_minus_1 + 1 == id) {
            FLAMEGPU->setVariable<int>("result1", 1);
        }
        // Attempting to iterate edges does not cause a crash (mostly a seatbelts test)
        const int result2 = graph.inEdges(FLAMEGPU->getIndex()).size();
        FLAMEGPU->setVariable<int>("result2", result2);
        const int result3 = graph.outEdges(FLAMEGPU->getIndex()).size();
        FLAMEGPU->setVariable<int>("result3", result3);
        return flamegpu::ALIVE;   
    }
    """
    
    def test_NoEdgesValid(self):
        # Graph can have no edges
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");
        graph.newVertexPropertyInt("foo");
        agent = model.newAgent("agent");
        agent.newVariableInt("result1", 0);
        agent.newVariableInt("result2", 1);
        agent.newVariableInt("result3", 1);
        a = agent.newRTCFunction("fn2", self.CheckGraphNoEdges_func);
        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_NoEdges());
        model.newLayer().addAgentFunction(a);

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        # Check results
        sim.getPopulationData(pop);

        for result_agent in pop:
            assert result_agent.getVariableInt("result1") == 1
            assert result_agent.getVariableInt("result2") == 0
            assert result_agent.getVariableInt("result3") == 0

    CheckGraphSameSrcDest_func = """
    FLAMEGPU_AGENT_FUNCTION(CheckGraphSameSrcDest, flamegpu::MessageNone, flamegpu::MessageNone) {
        auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
        // Can still access vertex data
        const unsigned int id = graph.getVertexID(FLAMEGPU->getIndex());
        const unsigned int id_minus_1 = graph.getVertexProperty<int>("foo", FLAMEGPU->getIndex());
        for (auto &edge : graph.inEdges(FLAMEGPU->getIndex())) {
            if (edge.getEdgeSource() == FLAMEGPU->getIndex()) {
                FLAMEGPU->setVariable<int>("result1", 1);
                if (edge.getIndex() == graph.getEdgeIndex(FLAMEGPU->getIndex(), FLAMEGPU->getIndex())) {
                    FLAMEGPU->setVariable<int>("result3", 1);
                }
            }
        }
        for (auto &edge : graph.outEdges(FLAMEGPU->getIndex())) {
            if (edge.getEdgeDestination() == FLAMEGPU->getIndex()) {
                FLAMEGPU->setVariable<int>("result2", 1);
            }
        }

        return flamegpu::ALIVE;
    }
    """
    
    def test_EdgeSameSrcDest(self):
        # Graph can have no edges
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");
        graph.newVertexPropertyInt("foo");
        graph.newEdgePropertyUInt("id");
        agent = model.newAgent("agent");
        agent.newVariableInt("result1", 0);
        agent.newVariableInt("result2", 0);
        agent.newVariableInt("result3", 0);
        a = agent.newRTCFunction("fn2", self.CheckGraphSameSrcDest_func);
        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_SameSrcDest());
        model.newLayer().addAgentFunction(a);

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        # Check results
        sim.getPopulationData(pop);

        for result_agent in pop:
            assert result_agent.getVariableInt("result1") == 1
            assert result_agent.getVariableInt("result2") == 1
            assert result_agent.getVariableInt("result3") == 1
            
          
    CheckGraphVertexFromID_func = """  
    FLAMEGPU_AGENT_FUNCTION(CheckGraphVertexFromID, flamegpu::MessageNone, flamegpu::MessageNone) {
        auto graph = FLAMEGPU->environment.getDirectedGraph("graph");

        const unsigned int vertex_index = graph.getVertexIndex(FLAMEGPU->getIndex() + 1);
        const unsigned int vertex_val = graph.getVertexProperty<int>("foo", vertex_index);

        if (vertex_val == FLAMEGPU->getIndex()) {
            FLAMEGPU->setVariable<int>("result1", 1);
        }

        return flamegpu::ALIVE;
    }
    """
    
    def test_VertexFromID(self):
        # Graph can have no edges
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");
        graph.newVertexPropertyInt("foo");
        graph.newEdgePropertyUInt("id");
        agent = model.newAgent("agent");
        agent.newVariableInt("result1", 0);
        a = agent.newRTCFunction("fn2", self.CheckGraphVertexFromID_func);
        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_SameSrcDest());
        model.newLayer().addAgentFunction(a);

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        # Check results
        sim.getPopulationData(pop);

        for result_agent in pop:
            assert result_agent.getVariableInt("result1") == 1
    
    CheckGraphEdgeFromIDs_func = """ 
    FLAMEGPU_AGENT_FUNCTION(CheckGraphEdgeFromIDs, flamegpu::MessageNone, flamegpu::MessageNone) {
        auto graph = FLAMEGPU->environment.getDirectedGraph("graph");

        const unsigned int edge_index = graph.getEdgeIndex(FLAMEGPU->getIndex(), FLAMEGPU->getIndex());
        const unsigned int edge_val = graph.getEdgeProperty<unsigned int>("id", edge_index);

        if (edge_val == graph.getVertexID(FLAMEGPU->getIndex()) - 1) {
            FLAMEGPU->setVariable<int>("result1", 1);
        }

        return flamegpu::ALIVE;
    }
    """
    
    def test_EdgeFromIDs(self):
        # Graph can have no edges
        model = pyflamegpu.ModelDescription("GraphTest");
        graph = model.Environment().newDirectedGraph("graph");
        graph.newVertexPropertyInt("foo");
        graph.newEdgePropertyUInt("id");
        agent = model.newAgent("agent");
        agent.newVariableInt("result1", 0);
        a = agent.newRTCFunction("fn2", self.CheckGraphEdgeFromIDs_func);
        # Init graph with known data
        model.newLayer().addHostFunction(InitGraph_SameSrcDest());
        model.newLayer().addAgentFunction(a);

        # Each agent checks 1 ID
        pop = pyflamegpu.AgentVector(agent, ID_AGENT_COUNT);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop);

        sim.step();

        # Check results
        sim.getPopulationData(pop);

        for result_agent in pop:
            assert result_agent.getVariableInt("result1") == 1
