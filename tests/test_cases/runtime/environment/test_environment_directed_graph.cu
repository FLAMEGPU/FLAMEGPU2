/**
* Tests of EnvironmentDirectedGraph
*
* This could perhaps be split into multiple files, one per useful class (Description, Host, Device)
*/
#include <filesystem>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace test_environment_directed_graph {
FLAMEGPU_HOST_FUNCTION(InitGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(10);
    auto vertices = graph.vertices();
    for (unsigned int i = 1; i < 11; ++i) {
        auto vertex = vertices[i];
        vertex.setProperty<float>("vertex_float", static_cast<float>(i));
        vertex.setProperty<double, 2>("vertex_double2", 0, static_cast<double>(i + 11));
        vertex.setProperty<double>("vertex_double2", 1, static_cast<double>(i + 21));
        vertex.setProperty<int, 3>("vertex_int3", {static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3)});
    }
    graph.setEdgeCount(20);
    auto edges = graph.edges();
    // Edge source dest pairs are carefully set to ensure the defined order matches the sorted order
    // Furthermore no edges have a matching source/dest (assuming vcount=10, ecount=20)
    for (unsigned int i = 0; i < 10; ++i) {
        auto edge = edges[{(i / 2) + 1, ((i + 2) % 10) + 1}];
        edge.setProperty<int>("edge_int", static_cast<int>(i + 70));
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(i + 61));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(i + 51));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43)});
    }
    for (unsigned int i = 10; i < 20; ++i) {
        auto edge = edges[{(i / 2) + 1, ((i + 6) % 10) + 1}];
        edge.setProperty<int>("edge_int", static_cast<int>(i + 70));
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(i + 61));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(i + 51));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43)});
    }
}
// Init's same as InitGraph, however fills the vertices/edges with zero
FLAMEGPU_HOST_FUNCTION(InitGraph2) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(10);
    auto vertices = graph.vertices();
    for (unsigned int i = 1; i < 11; ++i) {
        auto vertex = vertices[i];
        vertex.setProperty<float>("vertex_float", static_cast<float>(0));
        vertex.setProperty<double, 2>("vertex_double2", 0, static_cast<double>(0));
        vertex.setProperty<double>("vertex_double2", 1, static_cast<double>(0));
        vertex.setProperty<int, 3>("vertex_int3", { static_cast<int>(0), static_cast<int>(0) , static_cast<int>(0) });
    }
    graph.setEdgeCount(20);
    auto edges = graph.edges();
    for (unsigned int i = 0; i < 10; ++i) {
        auto edge = edges[{(i % 10) + 1, 1}];
        edge.setProperty<int>("edge_int", 0);
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(0));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(0));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(0), static_cast<float>(0) , static_cast<float>(0) });
    }
    for (unsigned int i = 10; i < 20; ++i) {
        auto edge = edges[{(i % 10) + 1, 3}];
        edge.setProperty<int>("edge_int", static_cast<int>(0));
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(0));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(0));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(0), static_cast<float>(0) , static_cast<float>(0) });
    }
}
// Alternate version to InitGraph
FLAMEGPU_HOST_FUNCTION(InitGraph3) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(30);
    auto vertices = graph.vertices();
    for (unsigned int i = 1; i < 31; ++i) {
        auto vertex = vertices[i];
        vertex.setProperty<float>("vertex_float", static_cast<float>(i));
        vertex.setProperty<double, 2>("vertex_double2", 0, static_cast<double>(i + 11));
        vertex.setProperty<double>("vertex_double2", 1, static_cast<double>(i + 21));
        vertex.setProperty<int, 3>("vertex_int3", { static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3) });
    }
    graph.setEdgeCount(60);
    auto edges = graph.edges();
    // Edge source dest pairs are carefully set to ensure the defined order matches the sorted order
    // Furthermore no edges have a matching source/dest (assuming vcount=30, ecount=60)
    for (unsigned int i = 0; i < 30; ++i) {
        auto edge = edges[{(i / 2) + 1, ((i + 2) % 30) + 1}];
        edge.setProperty<int>("edge_int", static_cast<int>(i + 70));
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(i + 61));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(i + 51));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(i + 41), static_cast<float>(i + 52) , static_cast<float>(i + 43) });
    }
    for (unsigned int i = 30; i < 60; ++i) {
        auto edge = edges[{(i / 2) + 1, ((i + 18) % 30) + 1}];
        edge.setProperty<int>("edge_int", static_cast<int>(i + 70));
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(i + 61));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(i + 51));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(i + 41), static_cast<float>(i + 52) , static_cast<float>(i + 43) });
    }
}
// Set graph, it assumes vertice/edge counts are correct
FLAMEGPU_HOST_FUNCTION(SetGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    auto vertices = graph.vertices();
    for (unsigned int i = 1; i < 11; ++i) {
        auto vertex = vertices[i];
        vertex.setProperty<float>("vertex_float", static_cast<float>(i));
        vertex.setProperty<double, 2>("vertex_double2", 0, static_cast<double>(i + 11));
        vertex.setProperty<double>("vertex_double2", 1, static_cast<double>(i + 21));
        vertex.setProperty<int, 3>("vertex_int3", { static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3) });
    }
    // Edge source dest pairs are carefully set to ensure the defined order matches the sorted order
    // Furthermore no edges have a matching source/dest (assuming vcount=10, ecount=20)
    auto edges = graph.edges();
    for (unsigned int i = 0; i < 10; ++i) {
        auto edge = edges[{(i % 10) + 1, 1}];
        edge.setDestinationVertexID(2);
        edge.setSourceVertexID(i + 1);
        edge.setProperty<int>("edge_int", static_cast<int>(i + 70));
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(i + 61));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(i + 51));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) });
    }
    for (unsigned int i = 10; i < 20; ++i) {
        auto edge = edges[{(i % 10) + 1, 3}];
        edge.setSourceDestinationVertexID(i - 9, 4);
        edge.setProperty<int>("edge_int", static_cast<int>(i + 70));
        edge.setProperty<double, 2>("edge_double2", 0, static_cast<double>(i + 61));
        edge.setProperty<double>("edge_double2", 1, static_cast<double>(i + 51));
        edge.setProperty<float, 3>("edge_float3", { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) });
    }
}
FLAMEGPU_HOST_FUNCTION(HostCheckSetGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Vertices
    EXPECT_EQ(graph.getVertexCount(), 10u);
    auto vertices = graph.vertices();
    for (unsigned int i = 1; i < 11; ++i) {
        auto vertex = vertices[i];
        EXPECT_EQ(vertex.getProperty<float>("vertex_float"), static_cast<float>(i));
        EXPECT_EQ((vertex.getProperty<double, 2>("vertex_double2", 0)), static_cast<double>(i + 11));
        EXPECT_EQ(vertex.getProperty<double>("vertex_double2", 1), static_cast<double>(i + 21));
        std::array<int, 3> result = { static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3) };
        EXPECT_EQ((vertex.getProperty<int, 3>("vertex_int3")), result);
    }
    // Edges
    EXPECT_EQ(graph.getEdgeCount(), 20u);
    auto edges = graph.edges();
    int j = 0;
    for (unsigned int i = 0; i < 10; ++i, j+=2) {
        auto edge = edges[{i + 1, 2}];
        EXPECT_EQ(edge.getSourceVertexID(), i + 1);
        EXPECT_EQ(edge.getDestinationVertexID(), 2u);
        EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>(i + 70));
        EXPECT_EQ((edge.getProperty<double, 2>("edge_double2", 0)), static_cast<double>(i + 61));
        EXPECT_EQ(edge.getProperty<double>("edge_double2", 1), static_cast<double>(i + 51));
        std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) };
        EXPECT_EQ((edge.getProperty<float, 3>("edge_float3")), result);
    }
    j = 1;
    for (unsigned int i = 10; i < 20; ++i, j+=2) {
        auto edge = edges[{ i - 9, 4u }];
        EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>(i + 70));
        EXPECT_EQ((edge.getProperty<double, 2>("edge_double2", 0)), static_cast<double>(i + 61));
        EXPECT_EQ(edge.getProperty<double>("edge_double2", 1), static_cast<double>(i + 51));
        const std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) };
        EXPECT_EQ((edge.getProperty<float, 3>("edge_float3")), result);
    }
}
FLAMEGPU_HOST_FUNCTION(HostCheckGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Vertices
    EXPECT_EQ(graph.getVertexCount(), 10u);
    auto vertices = graph.vertices();
    for (unsigned int i = 1; i < 11; ++i) {
        auto vertex = vertices[i];
        EXPECT_EQ(vertex.getProperty<float>("vertex_float"), static_cast<float>(i));
        EXPECT_EQ((vertex.getProperty<double, 2>("vertex_double2", 0)), static_cast<double>(i + 11));
        EXPECT_EQ(vertex.getProperty<double>("vertex_double2", 1), static_cast<double>(i + 21));
        std::array<int, 3> result = { static_cast<int>(i + 1), static_cast<int>(i + 2) , static_cast<int>(i + 3) };
        EXPECT_EQ((vertex.getProperty<int, 3>("vertex_int3")), result);
    }
    EXPECT_EQ(graph.getEdgeCount(), 20u);
    auto edges = graph.edges();
    for (unsigned int i = 0; i < 10; ++i) {
        auto edge = edges[{ (i / 2) + 1, ((i + 2) % 10) + 1 }];
        EXPECT_EQ(edge.getSourceVertexID(), (i / 2) + 1);
        EXPECT_EQ(edge.getDestinationVertexID(), ((i + 2) % 10) + 1);
        EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>(i + 70));
        EXPECT_EQ((edge.getProperty<double, 2>("edge_double2", 0)), static_cast<double>(i + 61));
        EXPECT_EQ(edge.getProperty<double>("edge_double2", 1), static_cast<double>(i + 51));
        std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) };
        EXPECT_EQ((edge.getProperty<float, 3>("edge_float3")), result);
    }
    for (unsigned int i = 10; i < 20; ++i) {
        auto edge = edges[{ (i / 2) + 1, ((i + 6) % 10) + 1 }];
        EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>(i + 70));
        EXPECT_EQ((edge.getProperty<double, 2>("edge_double2", 0)), static_cast<double>(i + 61));
        EXPECT_EQ(edge.getProperty<double>("edge_double2", 1), static_cast<double>(i + 51));
        const std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 42) , static_cast<float>(i + 43) };
        EXPECT_EQ((edge.getProperty<float, 3>("edge_float3")), result);
    }
}
// Equivalent version to HostCheckGraph but for InitGraph3
FLAMEGPU_HOST_FUNCTION(HostCheckGraph3) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Vertices
    EXPECT_EQ(graph.getVertexCount(), 30u);
    auto vertices = graph.vertices();
    for (unsigned int i = 1; i < 31; ++i) {
        auto vertex = vertices[i];
        EXPECT_EQ(vertex.getProperty<float>("vertex_float"), static_cast<float>(i));
        EXPECT_EQ((vertex.getProperty<double, 2>("vertex_double2", 0)), static_cast<double>(i + 11));
        EXPECT_EQ(vertex.getProperty<double>("vertex_double2", 1), static_cast<double>(i + 21));
        std::array<int, 3> result = { static_cast<int>(i + 1), static_cast<int>(i + 2), static_cast<int>(i + 3) };
        EXPECT_EQ((vertex.getProperty<int, 3>("vertex_int3")), result);
    }
    // Edges
    EXPECT_EQ(graph.getEdgeCount(), 60u);
    auto edges = graph.edges();
    for (unsigned int i = 0; i < 30; ++i) {
        auto edge = edges[{ (i / 2) + 1, ((i + 2) % 30) + 1 }];
        EXPECT_EQ(edge.getSourceVertexID(), (i / 2) + 1);
        EXPECT_EQ(edge.getDestinationVertexID(), ((i + 2) % 30) + 1);
        EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>(i + 70));
        EXPECT_EQ((edge.getProperty<double, 2>("edge_double2", 0)), static_cast<double>(i + 61));
        EXPECT_EQ(edge.getProperty<double>("edge_double2", 1), static_cast<double>(i + 51));
        std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 52), static_cast<float>(i + 43) };
        EXPECT_EQ((edge.getProperty<float, 3>("edge_float3")), result);
    }
    for (unsigned int i = 30; i < 60; ++i) {
        auto edge = edges[{ (i / 2) + 1, ((i + 18) % 30) + 1 }];
        EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>(i + 70));
        EXPECT_EQ((edge.getProperty<double, 2>("edge_double2", 0)), static_cast<double>(i + 61));
        EXPECT_EQ(edge.getProperty<double>("edge_double2", 1), static_cast<double>(i + 51));
        const std::array<float, 3> result = { static_cast<float>(i + 41), static_cast<float>(i + 52), static_cast<float>(i + 43) };
        EXPECT_EQ((edge.getProperty<float, 3>("edge_float3")), result);
    }
}
TEST(TestEnvironmentDirectedGraph, TestHostGetResetGet) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);
    // Check the data persists
    model.newLayer().addHostFunction(HostCheckGraph);
    // Init graph with different known data
    model.newLayer().addHostFunction(InitGraph3);
    // Check the data persists
    model.newLayer().addHostFunction(HostCheckGraph3);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, TestHostSetGet) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Init graph with junk data
    model.newLayer().addHostFunction(InitGraph2);
    // Set the graphs data to known data
    model.newLayer().addHostFunction(SetGraph);
    // Check the data persists
    model.newLayer().addHostFunction(HostCheckSetGraph);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_HOST_FUNCTION(HostException) {
    EXPECT_THROW(FLAMEGPU->environment.getDirectedGraph("does not exist"), exception::InvalidGraphName);
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");


    auto vertices = graph.vertices();
    auto edges = graph.edges();

    EXPECT_THROW(vertices[1], exception::OutOfBoundsException);
    EXPECT_THROW((edges[{1, 2}]), exception::OutOfBoundsException);

    EXPECT_NO_THROW(graph.setVertexCount(1));
    EXPECT_NO_THROW(graph.setEdgeCount(1));

    auto vertex = vertices[1];
    auto edge = edges[{1, 1}];

    // Name
    EXPECT_THROW(vertex.setProperty<float>("does not exist", static_cast<float>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.setProperty<double, 2>("does not exist", 0, static_cast<double>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(vertex.setProperty<double>("does not exist", 0, static_cast<double>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.setProperty<int, 3>("does not exist", { })), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.setProperty<int>("does not exist", static_cast<int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.setProperty<double, 2>("does not exist", 0, static_cast<double>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.setProperty<double>("does not exist", 0, static_cast<double>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.setProperty<float, 3>("does not exist", { })), exception::InvalidGraphProperty);
    EXPECT_THROW(vertex.getProperty<float>("does not exist"), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.getProperty<double, 2>("does not exist", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(vertex.getProperty<double>("does not exist", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.getProperty<int, 3>("does not exist")), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.getProperty<int>("does not exist"), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.getProperty<double, 2>("does not exist", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.getProperty<double>("does not exist", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.getProperty<float, 3>("does not exist")), exception::InvalidGraphProperty);

    // Type
    EXPECT_THROW(vertex.setProperty<unsigned int>("vertex_float", static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.setProperty<unsigned int, 2>("vertex_double2", 0, static_cast<unsigned int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(vertex.setProperty<unsigned int>("vertex_double2", 0, static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.setProperty<unsigned int, 3>("vertex_int3", { })), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.setProperty<unsigned int>("edge_int", static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.setProperty<unsigned int, 2>("edge_double2", 0, static_cast<unsigned int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.setProperty<unsigned int>("edge_double2", 0, static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.setProperty<unsigned int, 3>("edge_float3", { })), exception::InvalidGraphProperty);
    EXPECT_THROW(vertex.getProperty<unsigned int>("vertex_float"), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.getProperty<unsigned int, 2>("vertex_double2", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(vertex.getProperty<unsigned int>("vertex_double2", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.getProperty<unsigned int, 3>("vertex_int3")), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.getProperty<unsigned int>("edge_int"), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.getProperty<unsigned int, 2>("edge_double2", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.getProperty<unsigned int>("edge_double2", 0), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.getProperty<unsigned int, 3>("edge_float3")), exception::InvalidGraphProperty);

    // Length
    EXPECT_THROW(vertex.setProperty<int>("vertex_int3", static_cast<int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.setProperty<int, 2>("vertex_int3", 0, static_cast<int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.setProperty<int, 2>("vertex_int3", { })), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.setProperty<float>("edge_float3",  static_cast<unsigned int>(0)), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.setProperty<float, 2>("edge_float3", 0, static_cast<unsigned int>(0))), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.setProperty<float, 2>("edge_float3", { })), exception::InvalidGraphProperty);
    EXPECT_THROW(vertex.getProperty<int>("vertex_int3"), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.getProperty<int, 2>("vertex_int3", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW((vertex.getProperty<int, 2>("vertex_int3")), exception::InvalidGraphProperty);
    EXPECT_THROW(edge.getProperty<float>("edge_float3"), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.getProperty<float, 2>("edge_float3", 0)), exception::InvalidGraphProperty);
    EXPECT_THROW((edge.getProperty<float, 2>("edge_float3")), exception::InvalidGraphProperty);

    // Array Index
    EXPECT_THROW(vertex.setProperty<double>("vertex_double2", 3, static_cast<double>(0)), exception::OutOfBoundsException);
    EXPECT_THROW(edge.setProperty<double>("edge_double2", 3, static_cast<double>(0)), exception::OutOfBoundsException);
    EXPECT_THROW(vertex.getProperty<double>("vertex_double2", 3), exception::OutOfBoundsException);
    EXPECT_THROW(edge.getProperty<double>("edge_double2", 3), exception::OutOfBoundsException);
    EXPECT_THROW((vertex.setProperty<double, 2>("vertex_double2", 3, static_cast<double>(0))), exception::OutOfBoundsException);
    EXPECT_THROW((edge.setProperty<double, 2>("edge_double2", 3, static_cast<double>(0))), exception::OutOfBoundsException);
    EXPECT_THROW((vertex.getProperty<double, 2>("vertex_double2", 3)), exception::OutOfBoundsException);
    EXPECT_THROW((edge.getProperty<double, 2>("edge_double2", 3)), exception::OutOfBoundsException);

    // Vertex/Edge Index
    EXPECT_THROW(vertices[ID_NOT_SET], exception::IDOutOfBounds);
    EXPECT_THROW(vertex.setID(ID_NOT_SET), exception::IDOutOfBounds);
    EXPECT_THROW((edges[{ID_NOT_SET, 1}]), exception::IDOutOfBounds);
    EXPECT_THROW((edges[{1, ID_NOT_SET}]), exception::IDOutOfBounds);
    EXPECT_THROW(edge.setSourceVertexID(ID_NOT_SET), exception::IDOutOfBounds);
    EXPECT_THROW(edge.setDestinationVertexID(ID_NOT_SET), exception::IDOutOfBounds);
    EXPECT_THROW(edge.setSourceDestinationVertexID(ID_NOT_SET, 1), exception::IDOutOfBounds);
    EXPECT_THROW(edge.setSourceDestinationVertexID(1, ID_NOT_SET), exception::IDOutOfBounds);

    // Out of bounds
    EXPECT_THROW(vertices[2], exception::OutOfBoundsException);
}
TEST(TestEnvironmentDirectedGraph, TestHostException) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Init graph with junk data
    model.newLayer().addHostFunction(HostException);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent1, MessageNone, MessageNone) {
    if (FLAMEGPU->getID() <= 20) {
        DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
        if (FLAMEGPU->getID() <= 10) {
            FLAMEGPU->setVariable<id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
            // vertex_int3, device full array access not available, so skipped
        }
        FLAMEGPU->setVariable<id_t>("edge_source", graph.getVertexID(graph.getEdgeSource(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
        FLAMEGPU->setVariable<id_t>("edge_dest", graph.getVertexID(graph.getEdgeDestination(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
        FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
        // edge_float3, device full array access not available, so skipped
    }
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(CopyGraphToAgent3, MessageNone, MessageNone) {
    if (FLAMEGPU->getID() <= 60) {
        DeviceEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
        if (FLAMEGPU->getID() <= 30) {
            FLAMEGPU->setVariable<id_t>("vertex_id", graph.getVertexID(FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<float>("vertex_float", graph.getVertexProperty<float>("vertex_float", FLAMEGPU->getID() - 1));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 0, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 0));
            FLAMEGPU->setVariable<double, 2>("vertex_double2", 1, graph.getVertexProperty<double, 2>("vertex_double2", FLAMEGPU->getID() - 1, 1));
            // vertex_int3, device full array access not available, so skipped
        }
        FLAMEGPU->setVariable<id_t>("edge_source", graph.getVertexID(graph.getEdgeSource(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
        FLAMEGPU->setVariable<id_t>("edge_dest", graph.getVertexID(graph.getEdgeDestination(FLAMEGPU->getID() - 1)));  // Method returns index, convert back to ID
        FLAMEGPU->setVariable<int>("edge_int", graph.getEdgeProperty<int>("edge_int", FLAMEGPU->getID() - 1));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 0, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 0));
        FLAMEGPU->setVariable<double, 2>("edge_double2", 1, graph.getEdgeProperty<double, 2>("edge_double2", FLAMEGPU->getID() - 1, 1));
        // edge_float3, device full array access not available, so skipped
    }
    return flamegpu::ALIVE;
}
FLAMEGPU_HOST_FUNCTION(HostDeviceCheckGraph) {
    DeviceAgentVector agent = FLAMEGPU->agent("agent").getPopulationData();
    for (unsigned int i = 0; i < 20; ++i) {
        if (i < 10) {
            EXPECT_EQ(agent[i].getVariable<id_t>("vertex_id"), static_cast<id_t>((i + 1)));
            EXPECT_EQ(agent[i].getVariable<float>("vertex_float"), static_cast<float>(i + 1));
            std::array<double, 2> result = { static_cast<double>(i + 12), static_cast<double>(i + 22) };
            EXPECT_EQ((agent[i].getVariable<double, 2>("vertex_double2")), result);
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 2) % 10) + 1);
        } else {
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 6) % 10) + 1);
        }
        // Edges
        EXPECT_EQ(agent[i].getVariable<id_t>("edge_source"), static_cast<id_t>(i / 2) + 1);
        EXPECT_EQ(agent[i].getVariable<int>("edge_int"), static_cast<int>(i + 70));
        std::array<double, 2> result = { static_cast<double>(i + 61), static_cast<double>(i + 51) };
        EXPECT_EQ((agent[i].getVariable<double, 2>("edge_double2")), result);
    }
}
FLAMEGPU_HOST_FUNCTION(HostDeviceCheckGraph3) {
    DeviceAgentVector agent = FLAMEGPU->agent("agent").getPopulationData();
    for (unsigned int i = 0; i < 60; ++i) {
        if (i < 30) {
            EXPECT_EQ(agent[i].getVariable<id_t>("vertex_id"), static_cast<id_t>(i + 1));
            EXPECT_EQ(agent[i].getVariable<float>("vertex_float"), static_cast<float>(i + 1));
            std::array<double, 2> result = { static_cast<double>(i + 12), static_cast<double>(i + 22) };
            EXPECT_EQ((agent[i].getVariable<double, 2>("vertex_double2")), result);
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 2) % 30) + 1);
        } else {
            EXPECT_EQ(agent[i].getVariable<id_t>("edge_dest"), static_cast<id_t>((i + 18) % 30) + 1);
        }
        // Edges
        EXPECT_EQ(agent[i].getVariable<id_t>("edge_source"), static_cast<id_t>(i / 2) + 1);
        EXPECT_EQ(agent[i].getVariable<int>("edge_int"), static_cast<int>(i + 70));
        std::array<double, 2> result = { static_cast<double>(i + 61), static_cast<double>(i + 51) };
        EXPECT_EQ((agent[i].getVariable<double, 2>("edge_double2")), result);
    }
}
TEST(TestEnvironmentDirectedGraph, TestDeviceGetResetGet) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");

    agent.newVariable<id_t>("vertex_id");
    agent.newVariable<float>("vertex_float");
    agent.newVariable<double, 2>("vertex_double2");
    // agent.newVariable<int, 3>("vertex_int3");, device full array access not available, so skipped
    agent.newVariable<id_t>("edge_source");
    agent.newVariable<id_t>("edge_dest");
    agent.newVariable<int>("edge_int");
    agent.newVariable<double, 2>("edge_double2");
    // agent.newVariable<float, 3>("edge_float3");, device full array access not available, so skipped
    agent.newFunction("fn1", CopyGraphToAgent1);
    agent.newFunction("fn2", CopyGraphToAgent3);

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);
    // Copy Data from Graph to Agent
    model.newLayer().addAgentFunction(CopyGraphToAgent1);
    // Check the agent data is correct persists
    model.newLayer().addHostFunction(HostDeviceCheckGraph);
    // Init graph with different known data
    model.newLayer().addHostFunction(InitGraph3);
    // Copy Data from Graph to Agent
    model.newLayer().addAgentFunction(CopyGraphToAgent3);
    // Check the agent data is correct persists
    model.newLayer().addHostFunction(HostDeviceCheckGraph3);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(agent, 60);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_EXIT_CONDITION(always_exit) {
    return flamegpu::EXIT;
}
TEST(TestEnvironmentDirectedGraph, SubModelSet_MasterModelHostGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");
    submodel.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");

    // Init graph with known data
    submodel.newLayer().addHostFunction(InitGraph);
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();

    // Init graph with known data
    model.newLayer().addSubModel(sub_desc);

    // Check the data persists
    model.newLayer().addHostFunction(HostCheckGraph);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, SubModelSet_MasterModelDeviceGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");

    AgentDescription agent = submodel.newAgent("agent");
    agent.newVariable<float>("foobar");  // Agent is not required in this model for this test

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");

    submodel.newLayer().addHostFunction(InitGraph);

    // Copy data to agent and check in host fn
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription master_agent = model.newAgent("agent");
    master_agent.newVariable<float>("foobar");  // Agent is only used to init a population
    master_agent.newVariable<id_t>("vertex_id");
    master_agent.newVariable<float>("vertex_float");
    master_agent.newVariable<double, 2>("vertex_double2");
    // master_agent.newVariable<int, 3>("vertex_int3");, device full array access not available, so skipped
    master_agent.newVariable<id_t>("edge_source");
    master_agent.newVariable<id_t>("edge_dest");
    master_agent.newVariable<int>("edge_int");
    master_agent.newVariable<double, 2>("edge_double2");
    // master_agent.newVariable<float, 3>("edge_float3");, device full array access not available, so skipped

    master_agent.newFunction("fn1", CopyGraphToAgent1);

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();
    sub_desc.bindAgent("agent", "agent");

    // Init graph with known data
    model.newLayer().addSubModel(sub_desc);

    // Check the data persists
    model.newLayer().addAgentFunction(CopyGraphToAgent1);
    model.newLayer().addHostFunction(HostDeviceCheckGraph);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(master_agent, 20);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, MasterModelSet_SubModelHostGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");
    submodel.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");

    // Init graph with known data
    submodel.newLayer().addHostFunction(HostCheckGraph);
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    model.newAgent("agent").newVariable<float>("foobar");  // Agent is not used in this test

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);

    // Check the data persists
    model.newLayer().addSubModel(sub_desc);

    CUDASimulation sim(model);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, MasterModelSet_SubModelDeviceGet) {
    ModelDescription submodel("SubGraphTest");
    EnvironmentDirectedGraphDescription sub_graph = submodel.Environment().newDirectedGraph("graph");

    AgentDescription agent = submodel.newAgent("agent");
    agent.newVariable<id_t>("vertex_id");
    agent.newVariable<float>("vertex_float");
    agent.newVariable<double, 2>("vertex_double2");
    // agent.newVariable<int, 3>("vertex_int3");, device full array access not available, so skipped
    agent.newVariable<id_t>("edge_source");
    agent.newVariable<id_t>("edge_dest");
    agent.newVariable<int>("edge_int");
    agent.newVariable<double, 2>("edge_double2");
    // agent.newVariable<float, 3>("edge_float3");, device full array access not available, so skipped

    agent.newFunction("fn1", CopyGraphToAgent1);

    sub_graph.newVertexProperty<float>("vertex_float");
    sub_graph.newVertexProperty<double, 2>("vertex_double2");
    sub_graph.newVertexProperty<int, 3>("vertex_int3");

    sub_graph.newEdgeProperty<int>("edge_int");
    sub_graph.newEdgeProperty<double, 2>("edge_double2");
    sub_graph.newEdgeProperty<float, 3>("edge_float3");


    // Copy data to agent and check in host fn
    submodel.newLayer().addAgentFunction(CopyGraphToAgent1);
    submodel.newLayer().addHostFunction(HostDeviceCheckGraph);
    submodel.addExitCondition(always_exit);

    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription master_agent = model.newAgent("agent");
    master_agent.newVariable<float>("foobar");  // Agent is only used to init a population

    // Setup submodel
    SubModelDescription sub_desc = model.newSubModel("sub_graph", submodel);
    sub_desc.SubEnvironment().autoMapDirectedGraphs();
    sub_desc.bindAgent("agent", "agent");

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph);

    // Check the data persists
    model.newLayer().addSubModel(sub_desc);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(master_agent, 20);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
FLAMEGPU_HOST_FUNCTION(HostTestEdgesOut) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(5);
    graph.setEdgeCount(15);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (int i = 0; i < 5; ++i) {
        vertices[i + 1];
        for (int j = 0; j <= i; ++j) {
            auto edge = edges[{j + 1, i + 1}];
            edge.setProperty<id_t>("src_copy", j + 1);
        }
    }
}
FLAMEGPU_AGENT_FUNCTION(IterateEdgesOut, MessageNone, MessageNone) {
    id_t src = FLAMEGPU->getIndex();
    unsigned int ct = 0;
    bool src_all_correct = true;
    bool index_all_correct = true;
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
    auto filter = graph.outEdges(src);
    FLAMEGPU->setVariable<int>("count2", filter.size());
    for (auto &edge : filter) {
        src_all_correct &= edge.getProperty<id_t>("src_copy") == graph.getVertexID(src);
        index_all_correct &= edge.getIndex() == graph.getEdgeIndex(src, edge.getEdgeDestination());
        FLAMEGPU->setVariable<id_t, 5>("dests", ct, graph.getVertexID(edge.getEdgeDestination()));
        ++ct;
    }
    FLAMEGPU->setVariable<int>("count", ct);
    FLAMEGPU->setVariable<int>("src_all_correct", src_all_correct ? 1 : 0);
    FLAMEGPU->setVariable<int>("index_all_correct", index_all_correct ? 1 : 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_HOST_FUNCTION(HostTestEdgesIn) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(5);
    graph.setEdgeCount(15);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (int i = 0; i < 5; ++i) {
        vertices[i + 1];
        for (int j = 0; j <= i; ++j) {
            auto edge = edges[{i + 1, j + 1}];
            edge.setProperty<id_t>("dest_copy", j + 1);
        }
    }
}
FLAMEGPU_AGENT_FUNCTION(IterateEdgesIn, MessageNone, MessageNone) {
    id_t dest = FLAMEGPU->getIndex();
    unsigned int ct = 0;
    bool dest_all_correct = true;
    bool index_all_correct = true;
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
    auto filter = graph.inEdges(dest);
    FLAMEGPU->setVariable<int>("count2", filter.size());
    for (auto& edge : filter) {
        dest_all_correct &= edge.getProperty<id_t>("dest_copy") == graph.getVertexID(dest);
        index_all_correct &= edge.getIndex() == graph.getEdgeIndex(edge.getEdgeSource(), dest);
        FLAMEGPU->setVariable<id_t, 5>("srcs", ct, graph.getVertexID(edge.getEdgeSource()));

        ++ct;
    }
    FLAMEGPU->setVariable<int>("count", ct);
    FLAMEGPU->setVariable<int>("dest_all_correct", dest_all_correct ? 1 : 0);
    FLAMEGPU->setVariable<int>("index_all_correct", index_all_correct ? 1 : 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_HOST_FUNCTION(SaveGraph) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Save
    graph.exportGraph("graph.json");
}
FLAMEGPU_HOST_FUNCTION(LoadCheckGraph3) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Load
    graph.importGraph("graph.json");
    // Check
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 1; i < 31; ++i) {
        auto vertex = vertices[i];
        EXPECT_EQ(vertex.getProperty<float>("vertex_float"), static_cast<float>(i));
        EXPECT_EQ((vertex.getProperty<double, 2>("vertex_double2", 0)), static_cast<double>(i + 11));
        EXPECT_EQ(vertex.getProperty<double>("vertex_double2", 1), static_cast<double>(i + 21));
        std::array<int, 3> result = { static_cast<int>(i + 1), static_cast<int>(i + 2), static_cast<int>(i + 3) };
        EXPECT_EQ((vertex.getProperty<int, 3>("vertex_int3")), result);
    }

    for (unsigned int i = 1; i < 61; ++i) {
        if (i < 31) {
            auto vertex = vertices[i];
            EXPECT_EQ(vertex.getID(), static_cast<id_t>(i));
            EXPECT_EQ(vertex.getProperty<float>("vertex_float"), static_cast<float>(i));
            std::array<double, 2> result = { static_cast<double>(i + 11), static_cast<double>(i + 21) };
            EXPECT_EQ((vertex.getProperty<double, 2>)("vertex_double2"), result);
            auto edge = edges[{((i - 1) / 2) + 1, (((i - 1) + 2) % 30) + 1}];
            EXPECT_EQ(edge.getDestinationVertexID(), static_cast<id_t>(((i - 1) + 2) % 30) + 1);
            EXPECT_EQ(edge.getSourceVertexID(), static_cast<id_t>((i-1) / 2) + 1);
            EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>((i-1) + 70));
            std::array<double, 2> result2 = { static_cast<double>((i-1) + 61), static_cast<double>((i-1) + 51) };
            EXPECT_EQ((edge.getProperty<double, 2>)("edge_double2"), result2);
        } else {
            auto edge = edges[{((i-1) / 2) + 1, (((i-1) + 18) % 30) + 1}];
            EXPECT_EQ(edge.getDestinationVertexID(), static_cast<id_t>(((i-1) + 18) % 30) + 1);
            EXPECT_EQ(edge.getSourceVertexID(), static_cast<id_t>((i-1) / 2) + 1);
            EXPECT_EQ(edge.getProperty<int>("edge_int"), static_cast<int>((i-1) + 70));
            std::array<double, 2> result2 = { static_cast<double>((i-1) + 61), static_cast<double>((i-1) + 51) };
            EXPECT_EQ((edge.getProperty<double, 2>)("edge_double2"), result2);
        }
    }
    // Cleanup
    std::filesystem::remove("graph.json");
}
TEST(TestEnvironmentDirectedGraph, TestJSONSaveLoad) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph3);
    // Export
    model.newLayer().addHostFunction(SaveGraph);
    // Reinit graph with different data
    model.newLayer().addHostFunction(InitGraph);
    // Import graph and check it matches first init
    model.newLayer().addHostFunction(LoadCheckGraph3);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());
}
TEST(TestEnvironmentDirectedGraph, TestEdgesOut) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newEdgeProperty<id_t>("src_copy");

    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t, 5>("dests");
    agent.newVariable<int>("count");
    agent.newVariable<int>("count2");
    agent.newVariable<int>("src_all_correct");
    agent.newVariable<int>("index_all_correct");
    agent.newFunction("iterate_edges", IterateEdgesOut);

    // Init graph with known data
    model.newLayer().addHostFunction(HostTestEdgesOut);
    model.newLayer().addAgentFunction(IterateEdgesOut);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(agent, 5);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    sim.getPopulationData(pop);
    int k = 0;
    for (const auto& agt : pop) {
        EXPECT_EQ(agt.getVariable<int>("src_all_correct"), 1);
        EXPECT_EQ(agt.getVariable<int>("index_all_correct"), 1);
        EXPECT_EQ(agt.getVariable<int>("count"), 5 - k);
        EXPECT_EQ(agt.getVariable<int>("count2"), 5 - k);
        for (int i = 0; i < 5 - k; ++i) {
            EXPECT_EQ(agt.getVariable<id_t>("dests", i), static_cast<id_t>(k + i) + 1);
        }
        ++k;
    }
}
TEST(TestEnvironmentDirectedGraph, TestEdgesIn) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newEdgeProperty<id_t>("dest_copy");

    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t, 5>("srcs");
    agent.newVariable<int>("count");
    agent.newVariable<int>("count2");
    agent.newVariable<int>("dest_all_correct");
    agent.newVariable<int>("index_all_correct");
    agent.newFunction("iterate_edges", IterateEdgesIn);

    // Init graph with known data
    model.newLayer().addHostFunction(HostTestEdgesIn);
    model.newLayer().addAgentFunction(IterateEdgesIn);

    // Create enough agents, to copy all data from the 2nd graph init
    AgentVector pop(agent, 5);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    sim.getPopulationData(pop);
    int k = 0;
    for (const auto& agt : pop) {
        EXPECT_EQ(agt.getVariable<int>("dest_all_correct"), 1);
        EXPECT_EQ(agt.getVariable<int>("index_all_correct"), 1);
        EXPECT_EQ(agt.getVariable<int>("count"), 5 - k);
        EXPECT_EQ(agt.getVariable<int>("count2"), 5 - k);
        for (int i = 0; i < 5 - k; ++i) {
            EXPECT_EQ(agt.getVariable<id_t>("srcs", i), static_cast<id_t>(k + i) + 1);
        }
        ++k;
    }
}

const unsigned int ID_AGENT_COUNT = 1025;
const unsigned int ID_OFFSET = 29034;
const unsigned int ID_GAP1 = 3;
const unsigned int ID_GAP2 = 5;

FLAMEGPU_HOST_FUNCTION(InitGraph_ContiguousIDs) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int OFFSET = ID_OFFSET;
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT; ++i) {
        auto vertex = vertices[OFFSET + i];
        vertex.setProperty<unsigned int>("vertex_index", i);
        vertex.setProperty<unsigned int>("vertex_ID", OFFSET + i);
        // Test does not care about edges, but add some so that it generates properly
        edges[{OFFSET + i, OFFSET + ((i + 5) * 3) % ID_AGENT_COUNT}];
    }
}
FLAMEGPU_AGENT_FUNCTION(CheckGraph_ContiguousIDs, MessageNone, MessageNone) {
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
TEST(TestEnvironmentDirectedGraph, TestVertexIDContiguous) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<unsigned int>("vertex_index");
    graph.newVertexProperty<unsigned int>("vertex_ID");

    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("result1", 0);
    agent.newVariable<unsigned int>("result2", 0);
    agent.newVariable<unsigned int>("result3", 0);
    agent.newFunction("check_graph", CheckGraph_ContiguousIDs);

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_ContiguousIDs);
    model.newLayer().addAgentFunction(CheckGraph_ContiguousIDs);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    sim.getPopulationData(pop);
    for (const auto& agt : pop) {
        EXPECT_EQ(agt.getVariable<unsigned int>("result1"), 1u);
        EXPECT_EQ(agt.getVariable<unsigned int>("result2"), 1u);
        EXPECT_EQ(agt.getVariable<unsigned int>("result3"), 1u);
    }
}
FLAMEGPU_HOST_FUNCTION(InitGraph_NonContiguousIDs) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int OFFSET = ID_OFFSET;
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT; ++i) {
        const unsigned int my_id = OFFSET + i + ((i * ID_GAP2) - ((i * ID_GAP1) % ID_GAP2));
        auto vertex = vertices[my_id];
        vertex.setProperty<unsigned int>("vertex_index", i);
        vertex.setProperty<unsigned int>("vertex_ID", my_id);
        // Test does not care about edges, but add some so that it generates properly
        const unsigned int j = ((i + 5) * 3) % ID_AGENT_COUNT;
        const unsigned int my_dest_id = OFFSET + j + ((j * ID_GAP2) - ((j * ID_GAP1) % ID_GAP2));
        edges[{my_id, my_dest_id}];
    }
}
FLAMEGPU_AGENT_FUNCTION(CheckGraph_NonContiguousIDs, MessageNone, MessageNone) {
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
TEST(TestEnvironmentDirectedGraph, TestVertexIDNonContiguous) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<unsigned int>("vertex_index");
    graph.newVertexProperty<unsigned int>("vertex_ID");

    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("result1", 0);
    agent.newVariable<unsigned int>("result2", 0);
    agent.newVariable<unsigned int>("result3", 0);
    agent.newFunction("check_graph", CheckGraph_NonContiguousIDs);

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_NonContiguousIDs);
    model.newLayer().addAgentFunction(CheckGraph_NonContiguousIDs);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    sim.getPopulationData(pop);
    for (const auto& agt : pop) {
        EXPECT_EQ(agt.getVariable<unsigned int>("result1"), 1u);
        EXPECT_EQ(agt.getVariable<unsigned int>("result2"), 1u);
        EXPECT_EQ(agt.getVariable<unsigned int>("result3"), 1u);
    }
}
FLAMEGPU_HOST_FUNCTION(InitGraph_MissingIDs1) {
    // ID range < vertex count
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int OFFSET = ID_OFFSET;
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT/2; ++i) {
        const unsigned int my_id = OFFSET + i*2;
        auto vertex = vertices[my_id];
        // Test does not care about edges, but add some so that it generates properly
        const unsigned int j = i + 1 % (ID_AGENT_COUNT/2);
        edges[{my_id, OFFSET + j * 2}];
    }
}
TEST(TestEnvironmentDirectedGraph, TestAllVerticesRequireID1) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    AgentDescription agent = model.newAgent("agent");
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_MissingIDs1);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);
    EXPECT_THROW(sim.step(), exception::IDNotSet);
}
FLAMEGPU_HOST_FUNCTION(InitGraph_MissingIDs2) {
    // ID range > vertex count
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int OFFSET = ID_OFFSET;
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT/2; ++i) {
        const unsigned int my_id = OFFSET + i*5;
        auto vertex = vertices[my_id];
        // Test does not care about edges, but add some some so that it generates properly
        const unsigned int j = i + 1 % (ID_AGENT_COUNT/2);
        edges[{my_id, OFFSET + j * 5}];
    }
    vertices[OFFSET].setID(1);
}
TEST(TestEnvironmentDirectedGraph, TestAllVerticesRequireID2) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    AgentDescription agent = model.newAgent("agent");
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_MissingIDs2);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::IDNotSet);
}

FLAMEGPU_HOST_FUNCTION(InitGraph_InvalidEdgeSource) {
    // ID range > vertex count
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int OFFSET = ID_OFFSET;
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT; ++i) {
        const unsigned int my_id = OFFSET + i*5;
        auto vertex = vertices[my_id];
        const unsigned int j = i + 1 % (ID_AGENT_COUNT/2);
        edges[{my_id, OFFSET + j * 5}];
    }
    unsigned int i = ID_AGENT_COUNT / 2;
    const unsigned int my_id = OFFSET + i*5;
    const unsigned int j = i + 1 % (ID_AGENT_COUNT/2);
    auto edge = edges[{my_id, OFFSET + j * 5}];
    edge.setSourceVertexID(OFFSET - 1);
    edge.setDestinationVertexID(OFFSET);
}
FLAMEGPU_HOST_FUNCTION(InitGraph_InvalidEdgeDest) {
    // ID range > vertex count
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int OFFSET = ID_OFFSET;
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT; ++i) {
        const unsigned int my_id = OFFSET + i*5;
        auto vertex = vertices[my_id];
        const unsigned int j = i + 1 % (ID_AGENT_COUNT/2);
        edges[{my_id, OFFSET + j * 5}];
    }
    unsigned int i = ID_AGENT_COUNT / 2;
    const unsigned int my_id = OFFSET + i*5;
    const unsigned int j = i + 1 % (ID_AGENT_COUNT/2);
    auto edge = edges[{my_id, OFFSET + j * 5}];
    edge.setSourceVertexID(OFFSET);
    edge.setDestinationVertexID(OFFSET - 1);
}
TEST(TestEnvironmentDirectedGraph, TestInvalidEdgeSource) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    AgentDescription agent = model.newAgent("agent");
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_InvalidEdgeSource);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::InvalidID);
}
TEST(TestEnvironmentDirectedGraph, TestInvalidEdgeDest) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    AgentDescription agent = model.newAgent("agent");
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_InvalidEdgeDest);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::InvalidID);
}

FLAMEGPU_HOST_FUNCTION(InitGraph_NoEdges) {
    // ID range > vertex count
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(0);
    auto vertices = graph.vertices();
    for (unsigned int i = 0; i < ID_AGENT_COUNT; ++i) {
        auto vertex = vertices[i + 1];
        vertex.setProperty<int>("foo", i);
    }
}
FLAMEGPU_AGENT_FUNCTION(CheckGraphNoEdges, MessageNone, MessageNone) {
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
    // Can still access vertex data
    const unsigned int id = graph.getVertexID(FLAMEGPU->getIndex());
    const unsigned int id_minus_1 = graph.getVertexProperty<int>("foo", FLAMEGPU->getIndex());
    if (id_minus_1 + 1 == id) {
        FLAMEGPU->setVariable<int>("result1", 1);
    }
    // Attempting to iterate edges does not cause a crash (mostly a seatbelts test)
    const int result2 = graph.inEdges(FLAMEGPU->getIndex()).size() +  graph.outEdges(FLAMEGPU->getIndex()).size();
    FLAMEGPU->setVariable<int>("result2", result2);
    return flamegpu::ALIVE;
}
TEST(TestEnvironmentDirectedGraph, TestNoEdgesValid) {
    // Graph can have no edges
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");
    graph.newVertexProperty<int>("foo");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<int>("result1", 0);
    agent.newVariable<int>("result2", 1);
    agent.newFunction("fn2", CheckGraphNoEdges);
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_NoEdges);
    model.newLayer().addAgentFunction(CheckGraphNoEdges);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    // Check results
    sim.getPopulationData(pop);

    for (const auto &result_agent : pop) {
        EXPECT_EQ(result_agent.getVariable<int>("result1"), 1);
        EXPECT_EQ(result_agent.getVariable<int>("result2"), 0);
    }
}
FLAMEGPU_HOST_FUNCTION(InitGraph_SameSrcDest) {
    // ID range > vertex count
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT; ++i) {
        auto vertex = vertices[i + 1];
        vertex.setProperty<int>("foo", i);
        auto edge = edges[{i + 1, i + 1}];
        edge.setProperty<unsigned int>("id", i);
    }
}
FLAMEGPU_AGENT_FUNCTION(CheckGraphSameSrcDest, MessageNone, MessageNone) {
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
TEST(TestEnvironmentDirectedGraph, TestEdgeSameSrcDest) {
    // Graph can have no edges
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");
    graph.newVertexProperty<int>("foo");
    graph.newEdgeProperty<unsigned int>("id");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<int>("result1", 0);
    agent.newVariable<int>("result2", 0);
    agent.newVariable<int>("result3", 0);
    agent.newFunction("fn2", CheckGraphSameSrcDest);
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_SameSrcDest);
    model.newLayer().addAgentFunction(CheckGraphSameSrcDest);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    // Check results
    sim.getPopulationData(pop);

    for (const auto &result_agent : pop) {
        EXPECT_EQ(result_agent.getVariable<int>("result1"), 1);
        EXPECT_EQ(result_agent.getVariable<int>("result2"), 1);
        EXPECT_EQ(result_agent.getVariable<int>("result3"), 1);
    }
}
FLAMEGPU_AGENT_FUNCTION(CheckGraphVertexFromID, MessageNone, MessageNone) {
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");

    const unsigned int vertex_index = graph.getVertexIndex(FLAMEGPU->getIndex() + 1);
    const unsigned int vertex_val = graph.getVertexProperty<int>("foo", vertex_index);

    if (vertex_val == FLAMEGPU->getIndex()) {
        FLAMEGPU->setVariable<int>("result1", 1);
    }

    return flamegpu::ALIVE;
}
TEST(TestEnvironmentDirectedGraph, VertexFromID) {
    // Graph can have no edges
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");
    graph.newVertexProperty<int>("foo");
    graph.newEdgeProperty<unsigned int>("id");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<int>("result1", 0);
    agent.newFunction("fn2", CheckGraphVertexFromID);
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_SameSrcDest);
    model.newLayer().addAgentFunction(CheckGraphVertexFromID);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    // Check results
    sim.getPopulationData(pop);

    for (const auto &result_agent : pop) {
        EXPECT_EQ(result_agent.getVariable<int>("result1"), 1);
    }
}
FLAMEGPU_AGENT_FUNCTION(CheckGraphEdgeFromIDs, MessageNone, MessageNone) {
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int edge_index = graph.getEdgeIndex(FLAMEGPU->getIndex(), FLAMEGPU->getIndex());
    const unsigned int edge_val = graph.getEdgeProperty<unsigned int>("id", edge_index);

    if (edge_val == graph.getVertexID(FLAMEGPU->getIndex()) - 1) {
        FLAMEGPU->setVariable<int>("result1", 1);
    }

    return flamegpu::ALIVE;
}
TEST(TestEnvironmentDirectedGraph, EdgeFromIDs) {
    // Graph can have no edges
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");
    graph.newVertexProperty<int>("foo");
    graph.newEdgeProperty<unsigned int>("id");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<int>("result1", 0);
    agent.newFunction("fn2", CheckGraphEdgeFromIDs);
    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_SameSrcDest);
    model.newLayer().addAgentFunction(CheckGraphEdgeFromIDs);

    // Each agent checks 1 ID
    AgentVector pop(agent, ID_AGENT_COUNT);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_NO_THROW(sim.step());

    // Check results
    sim.getPopulationData(pop);

    for (const auto &result_agent : pop) {
        EXPECT_EQ(result_agent.getVariable<int>("result1"), 1);
    }
}

#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS1, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getVertexProperty<float>("vertex_float", 10);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS2, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getVertexProperty<double, 2>("vertex_double2", 10, 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS3, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getVertexID(10);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS1, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeProperty<int>("edge_int", 20);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS2, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeProperty<double, 2>("edge_double2", 20, 0);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS3, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeSource(20);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS4, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").getEdgeSource(20);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(DeviceOutEdges_SEATBELTS, MessageNone, MessageNone) {
    FLAMEGPU->environment.getDirectedGraph("graph").outEdges(10);
    return flamegpu::ALIVE;
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetVertex_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetVertex_SEATBELTS1);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetVertex_SEATBELTS1);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetVertex_SEATBELTS2) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetVertex_SEATBELTS2);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetVertex_SEATBELTS2);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetVertex_SEATBELTS3) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetVertex_SEATBELTS3);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetVertex_SEATBELTS3);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS1);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS1);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS2) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS2);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS2);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS3) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS3);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS3);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceGetEdge_SEATBELTS4) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceGetEdge_SEATBELTS4);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceGetEdge_SEATBELTS4);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_DeviceOutEdgesLeaving_SEATBELTS) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("fn1", DeviceOutEdges_SEATBELTS);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(DeviceOutEdges_SEATBELTS);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}

FLAMEGPU_HOST_FUNCTION(InitGraph_NonContiguousIDs_SEATBELTS) {
    HostEnvironmentDirectedGraph graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int OFFSET = ID_OFFSET;
    graph.setVertexCount(ID_AGENT_COUNT);
    graph.setEdgeCount(ID_AGENT_COUNT);
    auto vertices = graph.vertices();
    auto edges = graph.edges();
    for (unsigned int i = 0; i < ID_AGENT_COUNT; ++i) {
        const unsigned int my_id = OFFSET + (i * 2);
        auto vertex = vertices[my_id];
        // Test does not care about edges, but add some some so that it generates properly
        const unsigned int j = OFFSET + ((i + 1) % ID_AGENT_COUNT) * 2;
        edges[{my_id, j}];
    }
}
FLAMEGPU_AGENT_FUNCTION(ID_OutOfRangeGreater, MessageNone, MessageNone) {
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int my_vertex_index = graph.getVertexIndex(ID_OFFSET + (ID_AGENT_COUNT * 2) - 1);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ID_OutOfRangeLesser, MessageNone, MessageNone) {
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int my_vertex_index = graph.getVertexIndex(ID_OFFSET - 1);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ID_NotInUse, MessageNone, MessageNone) {
    auto graph = FLAMEGPU->environment.getDirectedGraph("graph");
    const unsigned int my_vertex_index = graph.getVertexIndex(ID_OFFSET + 1);
    return flamegpu::ALIVE;
}
TEST(TestEnvironmentDirectedGraph, Test_ID_OutOfRangeGreater_SEATBELTS) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("result1", 0);
    agent.newVariable<unsigned int>("result2", 0);
    agent.newVariable<unsigned int>("result3", 0);
    agent.newFunction("check_graph", ID_OutOfRangeGreater);

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_NonContiguousIDs_SEATBELTS);
    model.newLayer().addAgentFunction(ID_OutOfRangeGreater);

    // 1 Agent tests the exact bound
    AgentVector pop(agent, 1);
    CUDASimulation sim(model);
    sim.setPopulationData(pop);
    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_ID_OutOfRangeLesser_SEATBELTS) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("check_graph", ID_OutOfRangeLesser);

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_NonContiguousIDs_SEATBELTS);
    model.newLayer().addAgentFunction(ID_OutOfRangeLesser);

    // 1 Agent tests the exact bound
    AgentVector pop(agent, 1);
    CUDASimulation sim(model);
    sim.setPopulationData(pop);
    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph, Test_ID_NotInUse_SEATBELTS) {
    // Assign vertices an ID and check they are accessible correctly via map
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    AgentDescription agent = model.newAgent("agent");
    agent.newFunction("check_graph", ID_NotInUse);

    // Init graph with known data
    model.newLayer().addHostFunction(InitGraph_NonContiguousIDs_SEATBELTS);
    model.newLayer().addAgentFunction(ID_NotInUse);

    // 1 Agent tests the exact bound
    AgentVector pop(agent, 1);
    CUDASimulation sim(model);
    sim.setPopulationData(pop);
    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph_RTC, Test_DeviceGetVertex_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    const char *DeviceGetVertex_SEATBELTS1_RTC = R"###(
    FLAMEGPU_AGENT_FUNCTION(DeviceGetVertex_SEATBELTS1, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->environment.getDirectedGraph("graph").getVertexProperty<float>("vertex_float", 10);
        return flamegpu::ALIVE;
    }
    )###";
    auto t = agent.newRTCFunction("fn1", DeviceGetVertex_SEATBELTS1_RTC);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(t);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
TEST(TestEnvironmentDirectedGraph_RTC, Test_DeviceGetEdge_SEATBELTS1) {
    ModelDescription model("GraphTest");
    EnvironmentDirectedGraphDescription graph = model.Environment().newDirectedGraph("graph");

    graph.newVertexProperty<float>("vertex_float");
    graph.newVertexProperty<double, 2>("vertex_double2");
    graph.newVertexProperty<int, 3>("vertex_int3");

    graph.newEdgeProperty<int>("edge_int");
    graph.newEdgeProperty<double, 2>("edge_double2");
    graph.newEdgeProperty<float, 3>("edge_float3");

    AgentDescription agent = model.newAgent("agent");
    const char* DeviceGetEdge_SEATBELTS1_RTC = R"###(
    FLAMEGPU_AGENT_FUNCTION(DeviceGetEdge_SEATBELTS1, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->environment.getDirectedGraph("graph").getEdgeProperty<int>("edge_int", 20);
        return flamegpu::ALIVE;
    }
    )###";
    auto t = agent.newRTCFunction("fn1", DeviceGetEdge_SEATBELTS1_RTC);

    model.newLayer().addHostFunction(InitGraph);
    model.newLayer().addAgentFunction(t);

    AgentVector pop(agent, 1);

    CUDASimulation sim(model);
    sim.setPopulationData(pop);

    EXPECT_THROW(sim.step(), exception::DeviceError);
}
#else
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetVertex_SEATBELTS1) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetVertex_SEATBELTS2) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetVertex_SEATBELTS3) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS1) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS2) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS3) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceGetEdge_SEATBELTS4) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_DeviceOutEdgesLeaving_SEATBELTS) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_ID_OutOfRangeGreater_SEATBELTS) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_ID_OutOfRangeLesser_SEATBELTS) { }
TEST(TestEnvironmentDirectedGraph, DISABLED_Test_ID_NotInUse_SEATBELTS) { }
TEST(TestEnvironmentDirectedGraph_RTC, DISABLED_Test_DeviceGetVertex_SEATBELTS1) { }
TEST(TestEnvironmentDirectedGraph_RTC, DISABLED_Test_DeviceGetEdge_SEATBELTS1) { }
#endif

}  // namespace test_environment_directed_graph
}  // namespace flamegpu
