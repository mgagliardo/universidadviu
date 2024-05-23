import secrets
from datetime import datetime
from string import ascii_uppercase
from time import sleep


class Puerta:
    @staticmethod
    def esta_abierta():
        """
        Devuelve un booleano random que demuestra si la puerta se encuentra abierta o no
        """
        return bool(secrets.randbits(1))


class Vagon:
    puerta: Puerta
    numero_vagon: int

    def __init__(self, puerta: Puerta, numero_vagon: int):
        self.puerta = puerta
        self.numero_vagon = numero_vagon

    def get_numero_vagon(self) -> int:
        """
        Devuelve el numero del vagon
        """
        return self.numero_vagon

    def puerta_abierta(self) -> bool:
        """
        Devuelve el estado de la puerta:
            True = Abierta
            False = Cerrada
        """
        return self.puerta.esta_abierta()


class Tren:
    nombre: str
    vagones: list[Vagon]
    registro_paradas: dict = {}

    def _get_random_plate(self, len: int = 3, charsets: str = ascii_uppercase) -> str:
        csprng = secrets.SystemRandom()
        random_string = "".join(csprng.choices(charsets, k=len))
        random_int = secrets.randbelow(999)
        return f"{random_string}-{random_int}"

    def __init__(self, vagones: list[Vagon]):
        self.nombre = self._get_random_plate()
        self.vagones = vagones

    def get_nombre(self) -> str:
        return self.nombre

    def registrar_llegada(self, parada: str):
        """
        Registra el  horario actual en formato HH:MM:ss dd-mm-YYYY
        """
        horario_llegada = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        self.registro_paradas[parada] = horario_llegada
        print(
            f"Se registro una parada en la estacion {parada}. El horario de llegada es: {horario_llegada}"
        )

    def mostrar_horarios_llegada(self):
        """
        Imprime los horarios de llegada del tren
        """
        print("Mostrando a continuacion los horarios de llegada:\n")
        for parada, horario in self.registro_paradas.items():
            print(f"\t- Parada: {parada} - Horario: {horario}\n")

    def puede_arrancar(self) -> None:
        """ """
        # Demostracion de prueba para
        puerta_abierta = False
        for vagon in self.vagones:
            if vagon.puerta_abierta():
                print(
                    f"La puerta del vagon {vagon.get_numero_vagon()} se encuentra abierta, el tren {tren.get_nombre()} no puede arrancar\n"
                )
                puerta_abierta = True
                break

        if not puerta_abierta:
            print("No hay ninguna puerta abierta, el tren puede arrancar")


def probar_arranque(tren: Tren):
    tren.puede_arrancar()


def horarios_llegada(tren: Tren) -> None:
    """
    Funcion que muestra como se registra el horario de llegada y lo imprime al final
    """

    # Se demora 3 segundos entre cada registro para demostrar los diversos horarios de llegada
    tren.registrar_llegada("Alicante")
    sleep(3)

    tren.registrar_llegada("Valencia")
    sleep(3)

    tren.registrar_llegada("Barcelona")
    sleep(3)

    tren.registrar_llegada("Madrid")

    tren.mostrar_horarios_llegada()


if __name__ == "__main__":
    vagones = []
    cantidad_vagones = 10

    # Generando 10 vagones de prueba
    for numero_vagon in range(1, cantidad_vagones + 1):
        vagones.append(Vagon(Puerta(), numero_vagon))

    tren = Tren(vagones)

    # Funcion de prueba para verificar si un tren puede arrancar o no
    probar_arranque(tren)

    # Funcion de prueba para mostrar los horarios de registro y llegada de un tren
    horarios_llegada(tren)
