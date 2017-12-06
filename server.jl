

using HttpServer


http = HttpHandler() do req::Request, res::Response
    Response( ismatch(r"^/hello/",req.resource) ? string("Hello ", split(req.resource,'/')[3], "!") : 404 )
end

server = Server(http)
run(server,host=IPv4(127,0,0,1),port=8000)
