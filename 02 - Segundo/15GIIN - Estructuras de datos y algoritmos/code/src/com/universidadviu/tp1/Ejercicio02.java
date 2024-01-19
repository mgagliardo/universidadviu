package com.universidadviu.tp1;

class A {
    void foo() {
        System.out.println("Using XX");
    }

class A {
    ...
    A() {
        System.out.println("Constructing A");
        foo();
    }
}

class C extends A {
    C() {
        super.foo();
        System.out.println("Constructing C");
    }

    void foo() {
        System.out.println("Using C");
    }
}
