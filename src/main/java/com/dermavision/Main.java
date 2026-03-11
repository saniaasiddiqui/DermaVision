package com.dermavision;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.*;
import java.net.InetSocketAddress;

public class Main {
    public static void main(String[] args) throws Exception {

        HttpServer server = HttpServer.create(new InetSocketAddress(9090), 0);

        server.createContext("/", exchange -> {
            InputStream is = Main.class.getResourceAsStream("/static/index.html");
            if (is == null) {
                byte[] err = "index.html not found".getBytes();
                exchange.sendResponseHeaders(404, err.length);
                exchange.getResponseBody().write(err);
            } else {
                byte[] bytes = is.readAllBytes();
                exchange.getResponseHeaders().set("Content-Type", "text/html");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
            }
            exchange.getResponseBody().close();
        });

        server.start();
        System.out.println("DermaVision running at: http://localhost:9090");
    }
}