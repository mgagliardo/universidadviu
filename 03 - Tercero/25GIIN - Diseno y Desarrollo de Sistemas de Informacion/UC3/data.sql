-- Crea la base de datos
CREATE DATABASE IF NOT EXISTS ecoparque_db;
USE ecoparque_db;

-- Crea la tabla ecoparque
CREATE TABLE ecoparque (
    id_parque INT PRIMARY KEY,
    fecha_nombramiento DATE
);

-- Crea la tabla area
CREATE TABLE area (
    id_area INT PRIMARY KEY,
    nombre VARCHAR(255),
    extension DECIMAL(10, 2),
    id_parque INT,
    FOREIGN KEY (id_parque) REFERENCES ecoparque(id_parque)
);

-- Crea la tabla especie
CREATE TABLE especie (
    id_especie INT PRIMARY KEY,
    nombre_cientifico VARCHAR(255),
    nombre_vulgar VARCHAR(255),
    numero_individuos INT,
    id_area INT,
    FOREIGN KEY (id_area) REFERENCES area(id_area)
);

-- Crea la tabla especie_vegetal (hereda de especie)
CREATE TABLE especie_vegetal (
    id_especie INT PRIMARY KEY,
    flora BOOLEAN,
    periodo_floracion DATE,
    FOREIGN KEY (id_especie) REFERENCES especie(id_especie) ON DELETE CASCADE
);

-- Crea la tabla especie_animal (hereda de especie)
CREATE TABLE especie_animal (
    id_especie INT PRIMARY KEY,
    tipo_aliment ENUM('herbivora', 'carnivora', 'omnivora'),
    periodo_celo INT,
    FOREIGN KEY (id_especie) REFERENCES especie(id_especie) ON DELETE CASCADE
);

-- Crea la tabla especie_mineral (hereda de especie)
CREATE TABLE especie_mineral (
    id_especie INT PRIMARY KEY,
    tipo ENUM('roca', 'cristal'),
    FOREIGN KEY (id_especie) REFERENCES especie(id_especie) ON DELETE CASCADE
);

-- Crea la tabla alimentacion
CREATE TABLE alimentacion (
    depredador INT,
    presa INT,
    FOREIGN KEY (depredador) REFERENCES especie_animal(id_especie),
    FOREIGN KEY (presa) REFERENCES especie_animal(id_especie),
    PRIMARY KEY (depredador, presa)
);

-- Crea la tabla vehiculo
CREATE TABLE vehiculo (
    id_vehiculo INT PRIMARY KEY,
    matricula VARCHAR(10),
    tipo VARCHAR(50),
    area_asignada INT,
    FOREIGN KEY (area_asignada) REFERENCES area(id_area)
);

-- Crea la tabla personal
CREATE TABLE personal (
    dni INT PRIMARY KEY,
    nro_seg_social INT,
    nombre VARCHAR(255),
    direccion VARCHAR(255),
    telefono_dom INT,
    telefono_movil INT
);

-- Crea la tabla personal_gestion
CREATE TABLE personal_gestion (
    dni INT PRIMARY KEY, 
    id_entrada INT,
    FOREIGN KEY (dni) REFERENCES personal(dni) ON DELETE CASCADE
);

-- Crea la tabla personal_vigilancia
CREATE TABLE personal_vigilancia (
    dni INT PRIMARY KEY,
    id_vehiculo INT,
    id_area INT,
    FOREIGN KEY (id_vehiculo) REFERENCES vehiculo(id_vehiculo),
    FOREIGN KEY (id_area) REFERENCES area(id_area),
    FOREIGN KEY (dni) REFERENCES personal(dni) ON DELETE CASCADE
);

-- Crea la tabla personal_conservacion
CREATE TABLE personal_conservacion (
    dni INT PRIMARY KEY,
    id_area INT,
    especialidad VARCHAR(255),
    FOREIGN KEY (dni) REFERENCES personal(dni) ON DELETE CASCADE,
    FOREIGN KEY (dni) REFERENCES personal(dni),
    FOREIGN KEY (id_area) REFERENCES area(id_area)
);

-- Crea la tabla personal_investigador
CREATE TABLE personal_investigador (
    dni INT PRIMARY KEY,
    titulacion VARCHAR(255),
    FOREIGN KEY (dni) REFERENCES personal(dni) ON DELETE CASCADE,
    FOREIGN KEY (dni) REFERENCES personal(dni)
);

-- Crea la tabla proyecto
CREATE TABLE proyecto (
    id_proyecto INT PRIMARY KEY,
    presupuesto FLOAT,
    periodo INT,
    id_investigador INT,
    id_especie INT,
    FOREIGN KEY (id_investigador) REFERENCES personal_investigador(dni),
    FOREIGN KEY (id_especie) REFERENCES especie(id_especie)
);

-- Crea la tabla entrada
CREATE TABLE entrada (
    id_entrada INT PRIMARY KEY
);

-- Crea la tabla visitante
CREATE TABLE visitante (
    dni_vis INT PRIMARY KEY,
    nombre VARCHAR(255),
    domicilio VARCHAR(255),
    profesion VARCHAR(255)
);

-- Crea la tabla albergue
CREATE TABLE albergue (
    id_albergue INT PRIMARY KEY,
    nombre VARCHAR(255),
    capacidad INT,
    categoria VARCHAR(50)
);

-- Crea la tabla excursion
CREATE TABLE excursion (
    id_excursion INT PRIMARY KEY,
    transporte ENUM('vehiculo', 'a pie'),
    fecha DATE,
    hora TIME,
    id_albergue INT,
    FOREIGN KEY (id_albergue) REFERENCES albergue(id_albergue)
);

-- Crea la tabla de encuesta con id_excursion y id_albergue opcionales (nullable)
CREATE TABLE encuesta (
    id_encuesta INT AUTO_INCREMENT PRIMARY KEY,
    id_excursion INT,
    id_albergue INT,
    dni_vis INT,
    calificacion ENUM('1', '2', '3', '4', '5'),
    FOREIGN KEY (id_excursion) REFERENCES excursion(id_excursion) ON DELETE SET NULL,  -- FK nullable para excursion
    FOREIGN KEY (id_albergue) REFERENCES albergue(id_albergue) ON DELETE SET NULL,  -- FK nullable para albergue
    FOREIGN KEY (dni_vis) REFERENCES visitante(dni_vis)
);

---

INSERT INTO ecoparque (id_parque, fecha_nombramiento) VALUES
(1, '1985-06-15'); -- Un parque nombrado en 1985

-- Insertar areas
INSERT INTO area (id_area, nombre, extension, id_parque) VALUES
(1, 'Bosque de los alces', 5000.00, 1),
(2, 'Lago Encantado', 3000.50, 1),
(3, 'Piedra del aguila', 4000.75, 1),
(4, 'Selva Tropical', 7000.10, 1),
(5, 'Pradera de los ciervos', 6000.30, 1);

-- Inserta los datos en la tabla especie para especies vegetales
INSERT INTO especie (id_especie, nombre_cientifico, nombre_vulgar, numero_individuos, id_area) VALUES
(1, 'Pinus sylvestris', 'Pino silvestre', 500, 1),
(2, 'Salix babylonica', 'Sauce lloron', 350, 2),
(3, 'Quercus robur', 'Roble comun', 450, 3),
(4, 'Fagus sylvatica', 'Haya comun', 700, 1),
(5, 'Betula pendula', 'Abedul', 300, 4),
(6, 'Cedrus libani', 'Cedro del Libano', 250, 5),
(7, 'Fraxinus excelsior', 'Fresno comun', 800, 1),
(8, 'Acer pseudoplatanus', 'Arce', 400, 1),
(9, 'Alnus glutinosa', 'Aliso', 600, 1),
(10, 'Carya ovata', 'Nogal de nuez', 500, 1);

-- Inserta los datos en la tabla especie_vegetal (solo los atributos especificos)
INSERT INTO especie_vegetal (id_especie, flora, periodo_floracion) VALUES
(1, TRUE, '2023-04-15'), -- Pino silvestre
(2, TRUE, '2022-05-01'), -- Sauce lloron
(3, TRUE, '2021-05-20'), -- Roble comun
(4, TRUE, '2020-06-10'), -- Haya comun
(5, TRUE, '2022-04-30'), -- Abedul
(6, TRUE, '2021-06-15'), -- Cedro del Libano
(7, TRUE, '2019-05-25'), -- Fresno comun
(8, TRUE, '2023-04-05'), -- Arce
(9, TRUE, '2020-08-15'), -- Aliso
(10, TRUE, '2018-09-20'); -- Nogal de nuez

-- Inserta los datos en la tabla especie para especies animales
INSERT INTO especie (id_especie, nombre_cientifico, nombre_vulgar, numero_individuos, id_area) VALUES
(11, 'Panthera leo', 'Leon', 150,1),
(12, 'Canis lupus', 'Lobo', 80, 2),
(13, 'Ursus arctos', 'Oso pardo', 120, 3),
(14, 'Gorilla gorilla', 'Gorila', 50, 1),
(15, 'Elephas maximus', 'Elefante asiatico', 200, 4),
(16, 'Cervus elaphus', 'Ciervo rojo', 600, 5),
(17, 'Crocodylus porosus', 'Cocodrilo de agua salada', 20, 1),
(18, 'Hippopotamus amphibius', 'Hipopotamo', 30, 1),
(19, 'Acinonyx jubatus', 'Guepardo', 75, 1),
(20, 'Vulpes vulpes', 'Zorro rojo', 180, 1),
(21, 'Oncorhynchus nerka', 'Salmon Rojo', 35,  3);

-- Paso 2: Inserta los datos en la tabla especie_animal
INSERT INTO especie_animal (id_especie, tipo_aliment, periodo_celo) VALUES
(11, 'carnivora', 12),  -- Leon
(12, 'carnivora', 8),   -- Lobo
(13, 'omnivora', 6),   -- Oso pardo
(14, 'herbivora', 10),  -- Gorila
(15, 'herbivora', 24),  -- Elefante asiatico
(16, 'herbivora', 6),   -- Ciervo rojo
(17, 'carnivora', 6),   -- Cocodrilo de agua salada
(18, 'herbivora', 12),  -- Hipopotamo
(19, 'carnivora', 12),  -- Guepardo
(20, 'omnivora', 5),   -- Zorro rojo
(21, 'herbivora', 5);   -- Salmon Rojo

-- Paso 1: inserta los datos en la tabla especie para especies minerales
iNSERT iNTO especie (id_especie, nombre_cientifico, nombre_vulgar, numero_individuos, id_area) VALUES
(22, 'Basalto', 'Basalto', 150, 2),
(23, 'Caliza', 'Caliza', 200, 3),
(24, 'Cuarzo', 'Cuarzo', 300, 1),
(25, 'Amatista', 'Amatista', 250, 4),
(26, 'Marmol', 'Marmol', 100, 5),
(27, 'Pizarra', 'Pizarra', 50, 1),
(28, 'Diamante', 'Diamante', 10, 1),
(29, 'Topacio', 'Topacio', 30, 1),
(30, 'Roca sedimentaria', 'Roca sedimentaria', 200, 1),
(31, 'Granito', 'Granito', 100, 3);

-- Paso 3: inserta los datos en la tabla especie_mineral
iNSERT iNTO especie_mineral (id_especie, tipo) VALUES
(22, 'roca'),  -- Basalto
(23, 'roca'),  -- Caliza
(24, 'cristal'), -- Cuarzo
(25, 'cristal'), -- Amatista
(26, 'roca'),  -- Marmol
(27, 'roca'),  -- Pizarra
(28, 'cristal'), -- Diamante
(29, 'cristal'), -- Topacio
(30, 'roca'),  -- Roca sedimentaria
(31, 'roca');  -- Granito

-- Insertar Alimentacion
INSERT INTO alimentacion (depredador, presa) VALUES
(11, 19), -- El leon se alimenta del guepardo
(19, 21), -- El guepardo se alimenta del salmon
(13, 21), -- El oso se alimenta del salmon
(11, 16); -- El leon tambien se alimenta del ciervo

-- Insertar Vehiculos
INSERT INTO vehiculo (id_vehiculo, matricula, tipo, area_asignada) VALUES
(1, 'ABC1234', 'Camioneta', 1),
(2, 'XYZ5678', 'Auto', 2),
(3, 'LMN8901', 'Camioneta', NULL);

-- Insertar Personal
INSERT INTO personal (dni, nro_seg_social, nombre, direccion, telefono_dom, telefono_movil) VALUES
(12345678, 123456789, 'Juan Perez', 'Calle Ficticia 123', 555123456, 555987654),
(23456789, 234567890, 'Ana Lopez', 'Avenida Siempre Viva 456', 555234567, 555876543),
(34567890, 345678901, 'Carlos Garcia', 'Calle Los alamos 789', 555345678, 555765432),
(45678901, 456789012, 'Marta Rodriguez', 'Calle Larga 101', 555456789, 555654321),
(56789012, 567890123, 'Luis Fernandez', 'Avenida del Sol 202', 555567890, 555543210);

-- Insertar Personal de Gestion
INSERT INTO personal_gestion (dni, id_entrada) VALUES
(12345678, 1001),  -- Juan Perez
(23456789, 1002);  -- Ana Lopez

-- Insertar Personal de Vigilancia
INSERT INTO personal_vigilancia (dni, id_vehiculo, id_area) VALUES
(12345678, 1, 1),  -- Juan Perez con Vehiculo 1 en area 1
(23456789, 2, 2);  -- Ana Lopez con Vehiculo 2 en area 2

-- Insertar Personal de Conservacion
INSERT INTO personal_conservacion (dni, id_area, especialidad) VALUES
(34567890, 3, 'Ecologia de Montañas'),  -- Carlos Garcia en area 3
(45678901, 4, 'Conservacion de Selvas');  -- Marta Rodriguez en area 4

-- Insertar Personal Investigador
INSERT INTO personal_investigador (dni, titulacion) VALUES
(12345678, 'Licenciado en Biologia'),  -- Juan Perez
(23456789, 'Master en Ecologia'),     -- Ana Lopez
(34567890, 'Doctor en Ciencias Ambientales');  -- Carlos Garcia

-- Insertar Proyectos
INSERT INTO proyecto (id_proyecto, presupuesto, periodo, id_investigador, id_especie) VALUES
(1, 50000.00, 24, 12345678, 31),
(2, 35000.00, 12, 23456789, 20),
(3, 45000.00, 18, 34567890, 9),
(4, 15000.00, 6, 12345678, 4),
(5, 25000.00, 3, 23456789, 15),
(6, 35000.00, 12, 12345678, 22),
(7, 33000.00, 24, 23456789, 25);

-- Insertar Visitantes
INSERT INTO visitante (dni_vis, nombre, domicilio, profesion) VALUES
(98765432, 'Luis Gomez', 'Calle Real 101', 'Abogado'),
(87654321, 'Maria Lopez', 'Av. Santa Fe 303', 'Medico'),
(76543210, 'Juan Martinez', 'Calle Nueva 505', 'Ingeniero'),
(65432109, 'Laura Garcia', 'Callejon 202', 'Estudiante'),
(54321098, 'Pedro Sanchez', 'Avenida Central 101', 'Turista');

-- Insertar Albergues
INSERT INTO albergue (id_albergue, nombre, capacidad, categoria) VALUES
(1, "Casa del Sol", 50, 'Economico'),
(2, "Abadia", 30, 'Lujo'),
(3, "Cabana de Nahuel", 10, 'Economico'),
(4, "Hotel Hilton Bon Voy", 500, 'Lujo'),
(5, "Vinci Mae", 120, '3 Estrellas');

-- Insertar Excursiones
INSERT INTO excursion (id_excursion, transporte, fecha, hora, id_albergue) VALUES
(1, 'vehiculo', '2018-07-15', '10:00:00', 1),
(2, 'a pie', '2019-03-25', '14:00:00', 2),
(3, 'vehiculo', '2020-06-12', '08:30:00', 1);

-- Insertar Encuestas
INSERT INTO encuesta (id_encuesta, id_excursion, id_albergue, dni_vis, calificacion) VALUES
(1, 1, NULL, 98765432, '5'),
(2, 2, NULL, 87654321, '4'),
(3, 3, NULL, 76543210, '3'),
(4, 3, NULL, 65432109, '1'),
(5, 3, NULL, 54321098, '2');

INSERT INTO encuesta (id_encuesta, id_excursion, id_albergue, dni_vis, calificacion) VALUES
(6, NULL, 1, 98765432, '5'),
(7, NULL, 1, 87654321, '4'),
(8, NULL, 2, 76543210, '3'),
(9, NULL, 2, 65432109, '1'),
(10,NULL,  2, 54321098, '2');
