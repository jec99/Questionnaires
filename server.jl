

using HttpServer
import HttpServer.mimetypes
import HttpServer.FileResponse
using JSON
include("main.jl")

data = two_block(240,240,10);
points_tree,vecs_point,sensors_tree,vecs_sensor = organize(data,6,8);


http = HttpHandler() do req::Request, res::Response
    if req.resource == "/index"
        return FileResponse("index.html");
    elseif req.resource == "/matrix"
        mime = mimetypes["json"];
        s = JSON.json(data);
        return Response(200,Dict{AbstractString,AbstractString}([("Content-Type",mime)]),s);
    elseif req.resource == "/points"
        mime = mimetypes["json"];
        s = JSON.json(vecs_point[:,1:3]);
        return Response(200,Dict{AbstractString,AbstractString}([("Content-Type",mime)]),s);
    elseif req.resource == "/tree"
        mime = mimetypes["json"];
        s = JSON.json(as_dict(points_tree));
        return Response(200,Dict{AbstractString,AbstractString}([("Content-Type",mime)]),s);
    elseif req.resource == "/tree_example"
        # s = open(read,"flare.json");
        # mime = mimetypes["json"];
        # return Response(200, Dict{AbstractString,AbstractString}([("Content-Type",mime)]), s)
        return FileResponse("flare.json");
    elseif req.resource == "/matrix_example"
        mime = mimetypes["json"];
        A = rand(100,200);
        s = JSON.json(A);
        return Response(200, Dict{AbstractString,AbstractString}([("Content-Type",mime)]), s);
    elseif req.resource == "/points_example"
        mime = mimetypes["json"];
        A = rand(100,3);
        s = JSON.json(A);
        return Response(200, Dict{AbstractString,AbstractString}([("Content-Type",mime)]), s);
    else
        return Response(404);
    end
end

server = Server(http);
run(server,host=IPv4(127,0,0,1),port=8000);
